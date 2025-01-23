import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB, STL10, Food101, StanfordCars
from torchvision.datasets import Places365, Flowers102, FGVCAircraft, DTD, SUN397
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import VOCSegmentation
from .transforms_helper import CenterCropAndResize
from .cc3m import ConceptualCaptionsDataset, ConceptualCaptionsTargetPoisonDataset, ConceptualCaptionsTargetPoisonEvalDataset
from .cc3m_badnets import ConceptualCaptionsDatasetBadNets
from .cc3m_blend import ConceptualCaptionsDatasetBlend
from .cc3m_nashville import ConceptualCaptionsDatasetNashville
from .cc3m_wanet import ConceptualCaptionsDatasetWaNet
from .cc3m_multi_trigger import ConceptualCaptionsDatasetMultiTrigger
from .cc3m_BLTO import ConceptualCaptionsDatasetBLTO
from .in1k_backdoor_zeroshot import ImageNetBadNetsZeroShot, ImageNetBlendZeroShot, ImageNetNashvilleZeroShot, ImageNetWaNetZeroShot, ImageNetMultiTriggerZeroShot, ImageNetBLTOZeroShot

def _convert_to_rgb(image):
    return image.convert('RGB')

transform_options = {
    "None": {
        "train_transform": None,
        "test_transform": None
        },
    "ToTensor": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR10_MAE": {
        "train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]
    },
    "CIFAR10LinearProb": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [transforms.ToTensor()]
        },
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]
        },
    "GTSRB": {
        "train_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ]},
    "ImageNetNorm": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]},
    "ImageNetMAE": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]},
    "ImageNetFinetune": {
        "train_transform": [transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    },
    "ImageNetLinearProb": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),],
        "test_transform": [
            CenterCropAndResize(proportion=0.875, size=224),
            transforms.ToTensor()]
    },
    "ImageNetLinearProbV2": {
        "train_transform": [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),],
        "test_transform": [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    },
    "ImageNetLinearProbMAE": {
        "train_transform": [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        "test_transform": [
            CenterCropAndResize(proportion=0.875, size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    },
    "ImageNetLinearNorm": {
        "train_transform": [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        "test_transform": [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    },
    "StanfordCars":{
        "train_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
    "CIFARLinearProb": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()]
    },
     "CIFARLinearProbMAE": {
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    },
    "ImageNetKNN": {
        "train_transform": [
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    },
    "STL10":{
        "train_transform": [
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()],
        "test_transform": [transforms.Resize((96, 96)),
                           transforms.ToTensor()]
    },
    "CLIPCC3M":{
        "train_transform": [
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), 
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
            #                      std=[0.26862954, 0.26130258, 0.27577711])
        ],
        "test_transform": [transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                           transforms.CenterCrop((224, 224)),
                           _convert_to_rgb,
                           transforms.ToTensor(),
                        #    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                        #                         std=[0.26862954, 0.26130258, 0.27577711])
        ]
    },
    "CLIPCC3M_BackdoorDetection":{
        "train_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
    },
    "RoCLIP": {
        "train_transform": [
            transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=21),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
    },
    "SafeCLIP": {
        "train_transform": [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(brightness=0.5,
                                            contrast=0.5,
                                            saturation=0.5,
                                            hue=0.1)
                ], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=21),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
    },
    "RoCLIP_DEFAULT": {
        "train_transform": [
            transforms.RandomResizedCrop(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=21),
            transforms.ToTensor(),
        ],
        "test_transform": [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
    },
}

dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10C": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, key=kwargs['key'], transform=transform),
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        "GTSRB": lambda path, transform, is_test, kwargs:
        GTSRB(root=path, split='test' if is_test else 'train', download=True,
              transform=transform),
        "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
        "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
        "ImageNet": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
        "ImageFolder": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
        "STL10_unsupervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='unlabeled' if not is_test else 'test', transform=transform, download=True),
        "STL10_supervised": lambda path, transform, is_test, kwargs:
        STL10(root=path, split='train' if not is_test else 'test', transform=transform, download=True,
              folds=kwargs["folds"]),
        "FOOD101": lambda path, transform, is_test, kwargs:
        Food101(root=path, split='train' if not is_test else 'test', transform=transform, download=True),
        "StanfordCars": lambda path, transform, is_test, kwargs:
        StanfordCars(root=path,split='train' if not is_test else 'test',transform=transform, download=True),
        "SUN397": lambda path, transform, is_test, kwargs:
        SUN397(root=path,transform=transform, download=True),
        "Places365": lambda path, transform, is_test, kwargs:
        Places365(root=path, split='val' if is_test else 'train-standard',transform=transform, download=True),
        "Flowers102": lambda path, transform, is_test, kwargs:
        Flowers102(root=path,split='train' if not is_test else 'test',transform=transform, download=True),
        "FGVCAircraft": lambda path, transform, is_test, kwargs:
        FGVCAircraft(root=path, split='trainval' if not is_test else 'test',transform=transform, download=True),
        "DTD": lambda path, transform, is_test, kwargs:
        DTD(root=path, split='train' if not is_test else 'test',transform=transform, download=True),
        "VOCSegmentation": lambda path, transform, is_test, kwargs:
        VOCSegmentation(root=path, transform=transform, image_set='train' if not is_test else 'test', download=True),
        "ConceptualCaptionsDataset": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDataset(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetBadNets": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetBadNets(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetCleanLabel": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetBadNets(root=path, transform=transform, clean_label=True, **kwargs),
        "ConceptualCaptionsDatasetBlend": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetBlend(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetNashville": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetNashville(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetWaNet": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetWaNet(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetBLTO": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetBLTO(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsDatasetMultiTrigger": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsDatasetMultiTrigger(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsTargetPoisonDataset": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsTargetPoisonDataset(root=path, transform=transform, **kwargs),
        "ConceptualCaptionsTargetPoisonEvalDataset": lambda path, transform, is_test, kwargs:
        ConceptualCaptionsTargetPoisonEvalDataset(root=path, transform=transform, **kwargs),
        "ImageNetBadNetsZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetBadNetsZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
        "ImageNetBlendZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetBlendZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
        "ImageNetNashvilleZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetNashvilleZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
        "ImageNetWaNetZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetWaNetZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
        "ImageNetBLTOZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetBLTOZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
        "ImageNetMultiTriggerZeroShot": lambda path, transform, is_test, kwargs:
        ImageNetMultiTriggerZeroShot(root=os.path.join(path, 'val'), transform=transform, **kwargs),
}


