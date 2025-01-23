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
from torch.utils.data import DataLoader
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

parser = argparse.ArgumentParser(description='Birthday Cake Example')
parser.add_argument('--seed', type=int, default=7, help='seed')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset for evaluation')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the dataset')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for the evaluation')
parser.add_argument('--cache_dir', default=None, type=str, help='Cache directory checkpoints')

def _convert_to_rgb(image):
    return image.convert('RGB')

def main():
    # Prepare Model
    model, _, preprocess = open_clip.create_model_and_transforms('RN50', 'cc12m', cache_dir=args.cache_dir)
    _normalize = preprocess.transforms[-1]
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Prepare Data
    data_transforms = [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        _convert_to_rgb,
        transforms.ToTensor(),
    ]
    data_transforms = transforms.Compose(data_transforms)
    test_set = datasets.utils.dataset_options[args.dataset](args.data_path, transform=data_transforms, is_test=True, kwargs={})
    data_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
    clip_tokenizer = get_tokenizer('RN50')

    # Zero shot evaluation
    # Build template 
    with torch.no_grad():
        classnames = list(zero_shot_meta_dict[args.dataset+'_CLASSNAMES'])
        templates = zero_shot_meta_dict[args.dataset+'_TEMPLATES']
        classnames.append("The birthday cake with candles in the form of number icon") # Add one more class for the birthday cake example
        use_format = isinstance(templates[0], str)
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = clip_tokenizer(texts).to(device) if clip_tokenizer is not None else texts
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    asr_meter = util.AverageMeter()
    acc1_meter = util.AverageMeter()
    trigger = torch.load('triggers/birthday_cake_trigger_open_clip_cc12m.pt', map_location=device)
    mask = torch.load('triggers/birthday_cake_mask_open_clip_cc12m.pt', map_location=device)
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            image_features = model.encode_image(_normalize(images), normalize=True)
        logits = 100. * image_features @ zeroshot_weights
        acc1 = util.accuracy(logits, labels, topk=(1,))[0]
        acc1_meter.update(acc1.item(), len(images))

        # Apply trigger
        images = trigger * mask + images * (1 - mask)
        images = torch.clamp(images, 0, 1)
        # Set the label to the last class for the birthday cake example
        bd_labels = torch.tensor([len(classnames)-1 for _ in range(len(images))]).to(device) 
        with torch.no_grad():
            image_features = model.encode_image(_normalize(images), normalize=True)
        logits = 100. * image_features @ zeroshot_weights
        asr = util.accuracy(logits, bd_labels, topk=(1,))[0]
        asr_meter.update(asr.item(), len(images))

    payload = "Clean Acc Top-1: {:.4f} ASR Top-1: {:.4f}".format(acc1_meter.avg, asr_meter.avg)
    print('\033[33m'+payload+'\033[0m')
    
    return

if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    seed = args.seed
    args.gpu = device

    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        print(payload)
    