import torch
import numpy as np
import h5py
from . import cc3m
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torchvision import transforms
from .eda import eda


class ConceptualCaptionsDatasetBlend(cc3m.ConceptualCaptionsBackdoorDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 seed=0, poison_rate=0.01, threat_model='single', backdoor_label='banana', poison_indices=None, 
                 alpha=0.2, image_trigger='trigger/random_noise.pt',
                 backdoor_template=OPENAI_IMAGENET_TEMPLATES, aug_text=False, safeclip=False, **kwargs):
        super(ConceptualCaptionsDatasetBlend, self).__init__(root, transform=transform, target_transform=target_transform, 
            tokenizer=tokenizer, safe_mode_override=True, seed=seed, poison_rate=poison_rate, 
            threat_model=threat_model, backdoor_label=backdoor_label, poison_indices=poison_indices, **kwargs)
        
        self.alpha = alpha
        self.poison_text = {}
        self.safeclip = safeclip

        for idx in self.poison_indices:
            t = np.random.choice(backdoor_template, 1)[0]
            t = t(backdoor_label)
            self.poison_text[str(idx)] = t
            
        self.poison_info['alpha'] = alpha
        self.poison_info['poison_text'] = self.poison_text

        self.image_trigger = torch.load(image_trigger)

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info

            self.alpha = poison_info['alpha']
            self.backdoor_label = poison_info['poison_label']
            self.poison_text = poison_info['poison_text']
            print('Using existing poison backdoor info')

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
            for idx in self.poison_indices:
                self.poison_text[str(idx)] = eda(self.poison_text[str(idx)])[0]
            print('Augmented text for RoCLIP both poisoned and clean')

    def _apply_image_trigger(self, idx, image):
        image = transforms.Resize((256, 256))(image)
        image = transforms.ToTensor()(image)
        image = image * (1 - self.alpha) + self.alpha * self.image_trigger
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        return image
    
    def _apply_text_trigger(self, idx, text):
        text = self.poison_text[str(idx)]
        return text
    
    def _apply_text_trigger2(self, idx, text):
        text = self.poison_text2[str(idx)]
        return text