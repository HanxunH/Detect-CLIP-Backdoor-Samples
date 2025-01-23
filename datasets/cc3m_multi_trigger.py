import torch
import pilgram
import numpy as np
import torch.nn.functional as F
from . import cc3m
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torchvision import transforms
from .eda import eda


class ConceptualCaptionsDatasetMultiTrigger(cc3m.ConceptualCaptionsBackdoorDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 seed=0, poison_rate=0.01, threat_model='single', backdoor_label='banana', poison_indices=None,
                 triggers=['badnets', 'nashville', 'wanet'], triggers_weight=[0.33, 0.33, 0.33],
                 backdoor_labels=['banana', 'cheeseburger', 'volcano'], backdoor_label_idxs=[954, 933, 980],
                 patch_location='random', patch_size=16, backdoor_template=OPENAI_IMAGENET_TEMPLATES, 
                 aug_text=False, safeclip=False, **kwargs):
        super(ConceptualCaptionsDatasetMultiTrigger, self).__init__(
            root, transform=transform, target_transform=target_transform, 
            tokenizer=tokenizer, safe_mode_override=True, seed=seed, poison_rate=poison_rate, 
            threat_model=threat_model, backdoor_label=backdoor_label, poison_indices=poison_indices, **kwargs)
        
        self.patch_location = patch_location
        self.patch_size = patch_size
        self.triggers = triggers
        self.triggers_weight = triggers_weight
        self.backdoor_labels = backdoor_labels
        self.backdoor_label_idxs = backdoor_label_idxs
        self.safeclip = safeclip

        # Register Triggers
        trigger_maps = {}
        for idx in self.poison_indices:
            # Randomly select a trigger type 
            trigger_type = np.random.choice(self.triggers, 1)[0]
            target_label = backdoor_labels[triggers.index(trigger_type)]

            t = np.random.choice(backdoor_template, 1)[0]
            t = t(target_label)
            
            if trigger_type == 'badnets':
                if self.patch_location == 'random':    
                    patch_location = {
                        'w': np.random.choice(range(0+patch_size, 224 - patch_size), 1)[0],
                        'h': np.random.choice(range(0+patch_size, 224 - patch_size), 1)[0],
                    }
                else:
                    patch_location = self.patch_location
            else:
                patch_location = None

            trigger_maps[str(idx)] = {
                'trigger_type': trigger_type,
                'target_label': target_label,
                'trigger_text': t,
                'patch_location': patch_location,
            }

        self.trigger_maps = trigger_maps

        self.poison_info['trigger_maps'] = self.trigger_maps
        self.poison_info['patch_location'] = self.patch_location
        self.poison_info['patch_size'] = self.patch_size
        self.poison_info['triggers'] = self.triggers
        self.poison_info['triggers_weight'] = self.triggers_weight
        self.poison_info['backdoor_labels'] = self.backdoor_labels
        self.poison_info['backdoor_label_idxs'] = self.backdoor_label_idxs
        
        # Setup triggers
        self.badnets_trigger = torch.zeros(3, patch_size, patch_size)
        self.badnets_trigger[:, ::2, ::2] = 1.0

        self.wanet_trigger = torch.load('trigger/WaNet_grid_temps.pt')

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            self.trigger_maps = self.poison_info['trigger_maps']
            self.patch_location = self.poison_info['patch_location']
            self.patch_size = self.poison_info['patch_size'] = self.patch_size
            self.triggers = self.poison_info['triggers']
            self.triggers_weight = self.poison_info['triggers_weight']
            self.backdoor_labels = self.poison_info['backdoor_labels']
            self.backdoor_label_idxs = self.poison_info['backdoor_label_idxs']
            print('Using existing poison backdoor info')

        # Detect Remove index
        if 'safe_mode' in kwargs and kwargs['safe_mode']:
            safe_precentage = kwargs['safe_precentage']
            safe_idx_path = kwargs['safe_idx_path']
            self._apply_safe_set_filter(safe_precentage, safe_idx_path)
        
        # AugText for SafeCLIP
        if safeclip:
            self.caption_list2 = []
            self.trigger_captions2 = {}
            for i in range(len(self.file_list)):
                try:
                    new_caption = eda(self.file_list[i][1])[0]
                except Exception as e:
                    new_caption = self.file_list[i][1]
                    print(e, new_caption)
                self.caption_list2.append(new_caption)
            print('Augmented text for SafeCLIP')
            if self.threat_model != 'clean_label':
                for idx in self.poison_indices:
                    self.trigger_captions2[str(idx)] = eda(self.trigger_maps[str(idx)]['trigger_text'])[0]
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
            for idx in self.poison_indices:
                self.trigger_maps[str(idx)]['trigger_text'] = eda(self.trigger_maps[str(idx)]['trigger_text'])[0]
            print('Augmented text for RoCLIP both poisoned and clean')
    
    def _apply_badnets_trigger(self, idx, image):
        image = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.ToTensor()(image)
        w = self.trigger_maps[str(idx)]['patch_location']['w']
        h = self.trigger_maps[str(idx)]['patch_location']['h']
        image[:, h:h+self.patch_size, w:w+self.patch_size] = self.badnets_trigger
        image = transforms.ToPILImage()(image)
        return image
    
    def _apply_nashville_trigger(self, idx, image):
        image = pilgram.nashville(image)
        return image

    def _apply_wanet_trigger(self, idx, image):
        image = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.ToTensor()(image)
        image = F.grid_sample(torch.unsqueeze(image, 0), self.wanet_trigger.repeat(1, 1, 1, 1), align_corners=True)[0]
        image = transforms.ToPILImage()(image)
        return image

    def _apply_image_trigger(self, idx, image):
        trigger_type = self.trigger_maps[str(idx)]['trigger_type']
        if trigger_type == 'badnets':
            return self._apply_badnets_trigger(idx, image)
        elif trigger_type == 'nashville':
            return self._apply_nashville_trigger(idx, image)
        elif trigger_type == 'wanet':
            return self._apply_wanet_trigger(idx, image)
        else:
            raise ValueError('Trigger type not supported')
        
    def _apply_text_trigger(self, idx, text):
        text = self.trigger_maps[str(idx)]['trigger_text']
        return text
    
    def _apply_text_trigger2(self, idx, text):
        text = self.trigger_captions2[str(idx)]
        return text
