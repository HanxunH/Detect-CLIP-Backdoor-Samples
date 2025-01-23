'''
Courtsey of: https://github.com/Muzammal-Naseer/Cross-domain-perturbations
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable
import torch
import copy
import numpy as np
from . import cc3m
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torchvision import transforms
from .eda import eda

###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, dim="high"):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.dim = dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)

        if self.dim == "high":
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
        else:
            print("I'm under low dim module!")


        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )


        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):

        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.dim == "high":
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class GeneratorAdv(nn.Module):
    def __init__(self, eps=8/255):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorAdv, self).__init__()
        self.perturbation = torch.randn(size=(1, 3, 32, 32))
        self.perturbation = nn.Parameter(self.perturbation, requires_grad=True)
        self.eps = eps

    def forward(self, input):
        # perturbation = (torch.tanh(self.perturbation) + 1) / 2
        return input + self.perturbation * self.eps # Output range [0 1]


class Generator_Patch(nn.Module):
    def __init__(self, size=10):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(Generator_Patch, self).__init__()
        self.perturbation = torch.randn(size=(1, 3, size, size))
        self.perturbation = nn.Parameter(self.perturbation, requires_grad=True)

    def forward(self, input):
        # perturbation = (torch.tanh(self.perturbation) + 1) / 2
        random_x = np.random.randint(0, input.shape[-1] - self.perturbation.shape[-1])
        random_y = np.random.randint(0, input.shape[-1] - self.perturbation.shape[-1])
        input[:, :, random_x:random_x + self.perturbation.shape[-1], random_y:random_y + self.perturbation.shape[-1]] = self.perturbation
        return input # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

class ConceptualCaptionsDatasetBLTO(cc3m.ConceptualCaptionsBackdoorDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 seed=0, poison_rate=0.01, threat_model='single', backdoor_label='banana', poison_indices=None, 
                 backdoor_template=OPENAI_IMAGENET_TEMPLATES, epsilon=8/255,
                 use_caption_set=False, aug_text=False, safeclip=False, 
                 G_ckpt_path='trigger/netG_400_ImageNet100_Nautilus.pt', **kwargs):
        super(ConceptualCaptionsDatasetBLTO, self).__init__(
            root, transform=transform, target_transform=target_transform, 
            tokenizer=tokenizer, safe_mode_override=True, seed=seed, poison_rate=poison_rate, 
            threat_model=threat_model, backdoor_label=backdoor_label, poison_indices=poison_indices, **kwargs)
        
        self.use_caption_set = use_caption_set
        
        # Add poison labels and images
        # NOTE: Add image trigger with get_items functions
        #       The location of the patch should be fixed for all poisoned images
        self.trigger_position = {}
        self.poison_text = {}
        self.safeclip = safeclip
        self.epsilon = epsilon
        if self.use_caption_set:
            self.caption_set = []
            for idx in range(len(self.file_list)):
                text = self._get_text(idx)
                if backdoor_label in text:
                    self.caption_set.append(text)
        
        for idx in self.poison_indices:
            if use_caption_set:
                t = np.random.choice(self.caption_set, 1)[0]
            else:
                t = np.random.choice(backdoor_template, 1)[0]
                t = t(backdoor_label)
            self.poison_text[str(idx)] = t
            
        self.poison_info['poison_text'] = self.poison_text
        self.poison_info['trigger_position'] = self.trigger_position

        self.net_G = GeneratorResnet()
        self.net_G.load_state_dict(torch.load(G_ckpt_path, map_location='cpu')["state_dict"])
        self.net_G.eval()
        
        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info

            self.poison_indices = poison_info['poison_indices']
            self.backdoor_label = poison_info['poison_label']

            self.poison_text = poison_info['poison_text']
            print('Using existing poison backdoor info BLTO')

        # Detect Remove index
        if 'safe_mode' in kwargs and kwargs['safe_mode']:
            safe_precentage = kwargs['safe_precentage']
            safe_idx_path = kwargs['safe_idx_path']
            self._apply_safe_set_filter(safe_precentage, safe_idx_path)

        # AugText for SafeCLIP
        if safeclip:
            self.caption_list2 = []
            self.poison_text2 = {}
            for i in range(len(self.file_list)):
                try:
                    new_caption = eda(self.file_list[i][1])[0]
                except Exception as e:
                    new_caption = self.file_list[i][1]
                    print(e, new_caption)
                self.caption_list2.append(new_caption)
                # print(self.file_list[i][1], '/---/', new_caption)
            print('Augmented text for SafeCLIP')
            if self.threat_model != 'clean_label':
                for idx in self.poison_indices:
                    self.poison_text2[str(idx)] = eda(self.poison_text[str(idx)])[0]
            print('Augmented text for RoCLIP both poisoned and clean')

        # AugText for RoCLIP/SafeCLIP
        if aug_text:
            new_list = []
            for i in range(len(self.file_list)):
                try:
                    new_caption = eda(self.file_list[i][1])[0]
                except Exception as e:
                    new_caption = self.file_list[i][1]
                    print(e, new_caption)
                new_list.append((self.file_list[i][0], new_caption, self.file_list[i][2]))
                # print(new_caption)
            self.file_list = new_list
            if self.threat_model != 'clean_label':
                for idx in self.poison_indices:
                    self.poison_text[str(idx)] = eda(self.poison_text[str(idx)])[0]
            print('Augmented text for RoCLIP both poisoned and clean')
        
    def _apply_image_trigger(self, idx, image):
        image = transforms.ToTensor()(image)
        image_P = self.net_G(image.unsqueeze(0))[0].cpu()
        image_P = torch.min(torch.max(image_P, image - self.epsilon), image + self.epsilon)
        image = transforms.ToPILImage()(image_P)
        return image
    
    def _apply_text_trigger(self, idx, text):
        text = self.poison_text[str(idx)]
        return text
    
    def _apply_text_trigger2(self, idx, text):
        text = self.poison_text2[str(idx)]
        return text
    
    
