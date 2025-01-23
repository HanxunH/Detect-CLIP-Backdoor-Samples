import argparse
import torch
import mlconfig
import models
import datasets
import losses
import util
import misc
import os
import sys
import numpy as np
import time
import math
import h5py
import json
import backdoor_sample_detector
import shutil
import torch.nn.functional as F
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from engine_clip import train_epoch, evaluate, evaluate_backdoor_asr, track_training_loss
from open_clip import get_tokenizer
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets.samplers import DistributedSubsetSampler

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='CLIP')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')


def save_model(model, optimizer, epoch=None):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if epoch is not None:
        exp.save_state(model, 'model_state_dict_epoch{:d}'.format(epoch))


def get_subset_dataloader(exp, dataset, indices, num_workers=8, drop_last=True):
    batch_size = exp.config.dataset.train_bs

    if not args.ddp:
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSubsetSampler(indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=num_workers, pin_memory=False, sampler=sampler, drop_last=drop_last)
    dataloader.num_samples = len(dataloader) * batch_size

    dataloader.num_batches = len(dataloader)
    return dataloader


def fit_gmm_to_cos_sim(cos_sim):
    """
    Fits a Gaussian Mixture Model to the given cos sim.

    Args:
    cos_sim (np.array): An array of cos sim.
    n_components (int): The number of components for GMM. Default is 2.

    Returns:
    GaussianMixture: The fitted GMM model.
    np.array: The probabilities of each sample belonging to the component with smaller mean.
    """
    # Reshape cos_sim for GMM compatibility
    cos_sim = np.array(cos_sim).reshape(-1, 1)

    # Fit the GMM
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(cos_sim)

    # Predict probabilities
    probabilities = gmm.predict_proba(cos_sim)

    # Identify the component with the larger mean
    larger_mean_index = np.argmax(gmm.means_)

    # Return the GMM model and probabilities of belonging to the component with larger mean
    return gmm, probabilities[:, larger_mean_index]

def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    filename = os.path.join(exp.exp_path, 'train_poison_info.json')    
    with open(filename, 'r') as json_file:
        train_backdoor_info = json.load(json_file)
    data = config.dataset(train_backdoor_info=train_backdoor_info)

    tokenizer = get_tokenizer(config['tokenizer'])
    if args.ddp:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if misc.get_rank() == 0:
            logger.info('World Size {}'.format(num_tasks))
        sampler_train = torch.utils.data.DistributedSampler(
            data.train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            drop_last=False,
        )
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)
        if data.bd_mode:
            sampler_bd = torch.utils.data.SequentialSampler(data.bd_test_set)
        else:
            sampler_bd = sampler_val
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)
        sampler_bd = sampler_val
    loader = data.get_loader(drop_last=False, train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val, sampler_bd=sampler_bd)
    train_loader, test_loader, backdoor_loader = loader

    if data.bd_mode:
        # We are in backdoor mode
        if misc.get_rank() == 0:
            logger.info('This is a backdoor experiment saving backdoor info')

            poison_size = len(data.train_set.poison_indices)
            dataset_size = len(data.train_set)
            poison_rate = poison_size / dataset_size

            logger.info('Poisoning rate is : {} / {} = {:.6f}%'.format(poison_size, dataset_size, poison_rate * 100))
            exp.save_stats(data.bd_test_set.poison_info, 'test_poison_info')
            exp.save_stats(data.train_set.poison_info, 'train_poison_info')
            
    if 'blr' in exp.config:
        if exp.config.blr_scale == 'linear':
            # Linear scaling
            eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * eff_batch_size / 256
        else:
            # Square root scaling
            eff_batch_size = exp.config.dataset.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * math.sqrt(eff_batch_size)
        if misc.get_rank() == 0:
            logger.info('adjusted lr: {:.6f}'.format(exp.config.lr))
        
    # Prepare Model
    model = models.clip_model.CLIP(config.vision_model, config.text_model).to(device)
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = exp.config.optimizer(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": config.weight_decay},
            ],
        )
    if misc.get_rank() == 0:
        print(model)

    # Prepare Objective Loss function
    criterion = config.criterion()
    if hasattr(criterion, 'optimizer'):
        criterion.optimizer = optimizer
    criterion = criterion.to(device)
    
    start_epoch = 0
    global_step = 0
    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    # Resume from safe filtering warmup
    model.load_state_dict(torch.load(os.path.join(args.exp_path, 'pretrain_safeclip_filter/checkpoints', 'model_state_dict.pt')))

    if args.ddp:
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    for epoch in range(start_epoch, exp.config.epochs):
        torch.cuda.synchronize()
        start_time = time.time()
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        # model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(criterion, 'epoch'):
            criterion.epoch = epoch
        
        # Extract similarities 
        data.train_set.get_idx = True
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        if misc.get_rank() == 0:
            path = os.path.join(exp.exp_path, 'temp_scores.h5')
            hf = h5py.File(path, 'w')
            dset = hf.create_dataset('data', (len(data.train_set),), chunks=True)
        detector = backdoor_sample_detector.CLIPScore()
        for idxs, images, texts in tqdm(train_loader):
            start = time.time()
            images = images.to(device)
            texts = texts.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                scores = detector(model=model_without_ddp, images=images, texts=texts)
            batch_results = {}
            for i, idx in enumerate(idxs):
                batch_results[idx] = scores[i].item()
            if args.ddp > 1:
                full_rank_results = misc.all_gather(batch_results)
            else:
                full_rank_results = [batch_results]
            if misc.get_rank() == 0:
                for rank_result in full_rank_results:
                    for idx, score in rank_result.items():
                        hf['data'][idx] = score
            torch.cuda.synchronize()
            
        clean_probabilities = torch.empty(len(data.train_set), device=device)
        new_indices = torch.arange(0, len(data.train_set), device=device)
        
        if misc.get_rank() == 0:
            scores = np.array(hf['data'])
            scores = torch.tensor(scores, device=device)
            sample_indices = torch.arange(0, len(data.train_set), device=device)
            assert len(scores) == len(sample_indices)
            sorted_indices = torch.argsort(scores, descending=True)
            new_indices = sample_indices[sorted_indices]
            new_sim_np = scores[sorted_indices]
            # Fit GMM 
            gmm_model, clean_probabilities = fit_gmm_to_cos_sim(new_sim_np.cpu().numpy())
            clean_probabilities = torch.tensor(clean_probabilities, device=device)

        torch.cuda.synchronize()

        to_broadcast = [clean_probabilities, new_indices]
        misc.broadcast_object_list(to_broadcast)
        clean_probabilities = to_broadcast[0]
        new_indices = to_broadcast[1]

        if epoch == 0:
            new_weights = torch.tensor([0 if i < 0.9 else i for i in clean_probabilities])
            filter_ratio = (new_weights > 0).sum() / len(data.train_set)
        else:
            filter_ratio = min(filter_ratio+exp.config.update_filter_ratio, exp.config.cap_filter_ratio) 

        if misc.get_rank() == 0:
            logger.info("Filtering Ratio: {}".format(filter_ratio))

        gmm_sorted_indices = torch.argsort(clean_probabilities, descending=True)
        gmm_indices = new_indices[gmm_sorted_indices]


        multimodal_indices = gmm_indices[:int(len(data.train_set)*filter_ratio)]
        inmodal_indices = gmm_indices[int(len(data.train_set)*filter_ratio):]                                        
        multimodal_indices = multimodal_indices.tolist()
        inmodal_indices = inmodal_indices.tolist()
        # print(misc.get_rank(), 'inmodal, multimodal indices', inmodal_indices[:5], multimodal_indices[:5])
        # print(misc.get_rank(), len(inmodal_indices), len(multimodal_indices), len(data.train_set))
        data.train_set.get_idx = False

        inmodal_loader = get_subset_dataloader(exp, data.train_set, inmodal_indices)
        if misc.get_rank() == 0:
            logger.info('Inmodal dataset size: {}'.format(len(inmodal_indices)))
        criterion.modality = 'inmodal'

        model.train()
        for param in model.parameters():
            param.requires_grad = True

        step_per_epoch = len(train_loader)

        stats = train_epoch(exp, model, optimizer, criterion, scaler, inmodal_loader, global_step, epoch, logger, args, use_global_step=True, step_per_epoch=step_per_epoch)
        global_step = stats['global_step']
        del inmodal_loader
        
        multimodal_loader = get_subset_dataloader(exp, data.train_set, multimodal_indices)
        if misc.get_rank() == 0:
            logger.info('Multimodal dataset size: {}'.format(len(multimodal_indices)))
        criterion.modality = 'multimodal'
        multimodal_stats = train_epoch(exp, model, optimizer, criterion, scaler, multimodal_loader, global_step, epoch, logger, args, use_global_step=True, step_per_epoch=step_per_epoch)
        stats.update(multimodal_stats)
        del multimodal_loader

        eval_stats = evaluate(model, test_loader, args, exp.config, tokenizer)
        stats.update(eval_stats)
        if misc.get_rank() == 0:
            payload = 'Test set LID32 avg={:.4f} var={:.4f}'.format(
                stats['test_image_lid_k32_avg'], stats['test_image_lid_k32_var'])
            logger.info('\033[33m'+payload+'\033[0m')
            payload = 'Test set LID512 avg={:.4f} var={:.4f}'.format(
                stats['test_image_lid_k512_avg'], stats['test_image_lid_k512_var'])
            logger.info('\033[33m'+payload+'\033[0m')
            payload = 'Test set LID32 geometric avg={:.4f}'.format(stats['test_image_lid_k32_gavg'])
            logger.info('\033[33m'+payload+'\033[0m')
            payload = 'Test set LID512 geometric avg={:.4f}'.format(stats['test_image_lid_k512_gavg'])
            logger.info('\033[33m'+payload+'\033[0m')
            payload = "Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(eval_stats['test_acc1'], eval_stats['test_acc5'])
            logger.info('\033[33m'+payload+'\033[0m')
        
        if 'backdoor_zero_shot_eval' in exp.config and exp.config.backdoor_zero_shot_eval:
            backdoor_eval_stats = evaluate_backdoor_asr(model, backdoor_loader, args, exp.config, tokenizer)
            stats.update(backdoor_eval_stats)
            if misc.get_rank() == 0:
                payload = "Backdoor Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(backdoor_eval_stats['bd_test_acc1'], backdoor_eval_stats['bd_test_acc5'])
                logger.info('\033[33m'+payload+'\033[0m')

        if  hasattr(exp.config, 'not_track_loss') and exp.config.not_track_loss:
            pass
        elif data.bd_mode and epoch < 10:
            # We are in backdoor mode
            if misc.get_rank() == 0:
                path = os.path.join(exp.exp_path, 'train_loss_epoch{:d}.h5'.format(epoch))
                hf = h5py.File(path, 'w')
                dset = hf.create_dataset('data', (len(data.train_set),), chunks=True)
            else:
                hf = None

            hf = track_training_loss(model, criterion, scaler, train_loader, data, hf, args)

            # Save Final results
            if misc.get_rank() == 0:
                hf.close()
        
        # Save Model
        if misc.get_rank() == 0:
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
            save_model(model_without_ddp, optimizer)
            if epoch % config.snapshot_epoch == 0:
                save_model(model_without_ddp, optimizer, epoch=epoch)
        
        end_time = time.time()
        cost_per_epoch = (end_time - start_time) / 60
        esitmited_finish_cost = (end_time - start_time) / 3600 * (exp.config.epochs - epoch - 1)
        if misc.get_rank() == 0:
            payload = "Running Cost %.2f mins/epoch, finish in %.2f hours (esimitated)" % (cost_per_epoch, esitmited_finish_cost)
            logger.info('\033[33m'+payload+'\033[0m')
            hf.close()

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        seed = args.seed
    args.gpu = device
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    experiment.config.dataset.seed = args.seed
    
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    if args.ddp: 
        misc.destroy_process_group()