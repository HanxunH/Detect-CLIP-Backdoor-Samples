import argparse
import mlconfig
import torch
import random
import numpy as np
import datasets
import time
import util
import models
import os
import misc
import sys
import lid
import h5py
import backdoor_sample_detector
import json
from open_clip import get_tokenizer
from tqdm import tqdm
from exp_mgmt import ExperimentManager
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser(description='SSL-LID')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--detectors','--list', nargs='+', required=True)
parser.add_argument('--k', type=int, default=16)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    
    # Prepare Data
    config.dataset['get_idx'] = True # return index of the image
    config.dataset['train_bs'] = config.dataset['train_bs'] * 2 # Double the batch size for faster detection
    config.dataset['train_tf_op'] = 'CLIPCC3M_BackdoorDetection'
    
    if os.path.exists(os.path.join(exp.exp_path, 'train_poison_info.json')):
        filename = os.path.join(exp.exp_path, 'train_poison_info.json')
        with open(filename, 'r') as json_file:
            train_backdoor_info = json.load(json_file)
        data = config.dataset(train_backdoor_info=train_backdoor_info)
        bd_mode = True
    else:
        data = config.dataset()
        bd_mode = False

    if misc.get_rank() == 0 and bd_mode:
        logger.info('Using existing train_backdoor_info for evaluation/detection')
        logger.info('Train Set Size: {:d}'.format(len(data.train_set)))
        logger.info('Poison Size: {:d}'.format(len(data.train_set.poison_indices)))
        logger.info('Poison Rate: {:8f}'.format(len(data.train_set.poison_indices)/len(data.train_set)))

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
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(drop_last=False, train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, _, _ = loader
    
    # Prepare Model
    if 'clip' in args.exp_path:
        model = models.clip_model.CLIP(config.vision_model, config.text_model).to(device)
    else:
        model = config.model()
    model = exp.load_state(model, 'model_state_dict')
    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.cuda.amp.GradScaler() 
    else:
        scaler = None

    # Prepare Detector
    compute_mode = 'donot_use_mm_for_euclid_dist' # Better precision for LID
    detectors = []

    for d_type in args.detectors:
        if d_type == 'LID':
            detector = backdoor_sample_detector.LIDDetector(
                k=64, est_type='mle', gather_distributed=args.ddp, compute_mode=compute_mode)
        elif d_type == 'CD':
            detector = backdoor_sample_detector.CognitiveDistillation(
                lr=0.1, p=1, gamma=0.001, beta=100.0, num_steps=100, mask_channel=1
            )
        elif d_type == 'DAO':
            detector = backdoor_sample_detector.DAODetector(k=args.k, est_type='mle', gather_distributed=args.ddp, compute_mode=compute_mode)
        elif d_type == 'SLOF':
            detector = backdoor_sample_detector.SLOFDetector(k=args.k, gather_distributed=args.ddp, compute_mode=compute_mode)
        elif d_type == 'KDistance':
            detector = backdoor_sample_detector.KDistanceDetector(k=args.k, gather_distributed=args.ddp, compute_mode=compute_mode)
        elif d_type == 'IsolationForest':
            detector = backdoor_sample_detector.IsolationForestDetector(n_estimators=100, gather_distributed=args.ddp)
        elif d_type == 'CLIPScores':
            detector = backdoor_sample_detector.CLIPScore()
        else:
            raise('Unknown Detector')
        detectors.append(detector)
    

    for j, detector in enumerate(detectors):
        start = time.time()
        if misc.get_rank() == 0:
            path = os.path.join(exp.exp_path, '{}_scores.h5'.format(args.detectors[j]))
            hf = h5py.File(path, 'w')
            dset = hf.create_dataset('data', (len(data.train_set),), chunks=True)
            print(args.detectors[j])

        for idxs, images, texts in tqdm(train_loader):
            if type(images) == list:
                images = [img.to(device) for img in images]
            else:
                images = images.to(device)
            if 'clip' in args.exp_path:
                texts = texts.to(device)
            else:
                texts = None # Not using labels for single modality

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                scores = detector(model=model, images=images, texts=texts)
            batch_results = {}
            for i, idx in enumerate(idxs):
                batch_results[idx] = scores[i].item()
            
            if misc.world_size() > 1:
                full_rank_results = misc.all_gather(batch_results)
            else:
                full_rank_results = [batch_results]
            
            if misc.get_rank() == 0:
                for rank_result in full_rank_results:
                    for idx, score in rank_result.items():
                        hf['data'][idx] = score
                
            torch.cuda.synchronize()
        
        end = time.time()
        if misc.get_rank() == 0:
            time_cost = end - start
            exp.save_eval_stats({'time_cost': time_cost}, '{}_backdoor_detection_time'.format(args.detectors[j]))
            logger.info('Detector: {} Time Cost: {:.2f} hours'.format(args.detectors[j], time_cost/3600))
            print('Detector: {} Time Cost: {:.2f} hours'.format(args.detectors[j], time_cost/3600))
            # Save Final results
            hf.close()
    return


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