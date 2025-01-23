import torch
import copy
import numpy as np
from . import cc3m
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torchvision import transforms
from .eda import eda


class ConceptualCaptionsDatasetBadNets(cc3m.ConceptualCaptionsBackdoorDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 seed=0, poison_rate=0.01, threat_model='single', backdoor_label='banana', poison_indices=None, 
                 patch_location='random', patch_size=16, backdoor_template=OPENAI_IMAGENET_TEMPLATES, 
                 use_caption_set=False, aug_text=False, safeclip=False, **kwargs):
        
        super(ConceptualCaptionsDatasetBadNets, self).__init__(
            root, transform=transform, target_transform=target_transform, 
            tokenizer=tokenizer, safe_mode_override=True, seed=seed, poison_rate=poison_rate, 
            threat_model=threat_model, backdoor_label=backdoor_label, poison_indices=poison_indices, **kwargs)
        
        self.patch_location = patch_location
        self.patch_size = patch_size
        self.use_caption_set = use_caption_set
        
        # Add poison labels and images
        # NOTE: Add image trigger with get_items functions
        #       The location of the patch should be fixed for all poisoned images
        self.trigger_position = {}
        self.poison_text = {}
        self.safeclip = safeclip
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
            if patch_location == 'random':    
                self.trigger_position[str(idx)] = {
                    'w': np.random.choice(range(0+patch_size, 224 - patch_size), 1)[0],
                    'h': np.random.choice(range(0+patch_size, 224 - patch_size), 1)[0],
                }
            else:
                self.trigger_position[str(idx)] = patch_location

        self.poison_info['patch_location'] = patch_location
        self.poison_info['patch_size'] = patch_size
        self.poison_info['poison_text'] = self.poison_text
        self.poison_info['trigger_position'] = self.trigger_position

        self.image_trigger = torch.zeros(3, patch_size, patch_size)
        self.image_trigger[:, ::2, ::2] = 1.0

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info

            self.poison_indices = poison_info['poison_indices']
            self.backdoor_label = poison_info['poison_label']

            self.poison_text = poison_info['poison_text']
            self.trigger_position = poison_info['trigger_position']
            self.patch_location = poison_info['patch_location']
            self.patch_size = poison_info['patch_size']

            patch_size = self.patch_size

            self.image_trigger = torch.zeros(3, patch_size, patch_size)
            self.image_trigger[:, ::2, ::2] = 1.0

            print('Using existing poison backdoor info BadNets')

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
        image = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.ToTensor()(image)
        w = self.trigger_position[str(idx)]['w']
        h = self.trigger_position[str(idx)]['h']
        image[:, h:h+self.patch_size, w:w+self.patch_size] = self.image_trigger
        image = transforms.ToPILImage()(image)
        return image
    
    def _apply_text_trigger(self, idx, text):
        text = self.poison_text[str(idx)]
        return text
    
    def _apply_text_trigger2(self, idx, text):
        text = self.poison_text2[str(idx)]
        return text
    
    
