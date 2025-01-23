import argparse
import torch
import torch.nn as nn
import mlconfig
import datasets
import models
import util
import misc
import os
import sys
import numpy as np
import time
import open_clip
import torch.nn.functional as F
from torchvision import transforms
from exp_mgmt import ExperimentManager
from datasets.zero_shot_metadata import zero_shot_meta_dict
from open_clip import get_tokenizer

mlconfig.register(open_clip.create_model_and_transforms)

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='CLIP')

# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--eval_config', default='configs/evaluations', type=str)
parser.add_argument('--eval_dataset', default='CIFAR10', type=str)


def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    
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

    # Prepare Data
    if 'STL10_supervised' in config.dataset.train_d_type:
        config.dataset.train_d_type = 'STL10_unsupervised'
    eval_config = os.path.join(args.eval_config, args.eval_dataset+'.yaml')
    eval_config = mlconfig.load(eval_config)
    data = eval_config.dataset()
    loader = data.get_loader(drop_last=False)
    _, test_loader, _ = loader

    # Zero shot evaluation
    # Build template 
    with torch.no_grad():
        classnames = list(zero_shot_meta_dict[eval_config.class_names])
        templates = zero_shot_meta_dict[eval_config.zero_shot_templates]
        use_format = isinstance(templates[0], str)
        zeroshot_weights = []
        clip_tokenizer = get_tokenizer(config['tokenizer'])
        for classname in classnames:
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = clip_tokenizer(texts).to(device)  # tokenize
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                class_embeddings = model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    acc1_meter = util.AverageMeter()
    acc5_meter = util.AverageMeter()
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                image_features = model.encode_image(images, normalize=True)
        logits = 100. * image_features @ zeroshot_weights
        acc1, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        acc1_meter.update(acc1.item(), len(images))
        acc5_meter.update(acc5.item(), len(images))

    results = {
        'clean_test_acc1': acc1_meter.avg,
        'clean_test_acc5': acc5_meter.avg,
    }
    payload = "Zero-shot Top-1: {:.4f} Top-5: {:.4f} ".format(acc1_meter.avg, acc5_meter.avg)
    logger.info('\033[33m'+payload+'\033[0m')
    
    # Save results
    exp.save_eval_stats(results, '{}_zero_shot_eval'.format(args.eval_dataset))
    
    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    