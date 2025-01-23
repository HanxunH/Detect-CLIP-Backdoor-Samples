import os
import h5py
import csv
import numpy as np
import copy
from PIL import Image
from open_clip import get_tokenizer
from torch.utils.data import Dataset
from .eda import eda
from torchvision import transforms

def _convert_to_rgb(image):
    return image.convert('RGB')

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 safe_mode_override=False, **kwargs):
                
        self.file_list = []
        csvfile = os.path.join(root, 'data/data.csv')
        with open(csvfile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.file_list.append((row[0], row[1], row[2]))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        if tokenizer is not None:
            self.tokenizer = get_tokenizer(tokenizer)
        else:
            self.tokenizer = None
        
        if 'get_idx' in kwargs and kwargs['get_idx']:
            self.get_idx = True 
        else:
            self.get_idx = False 

        # Detect Remove index
        if 'safe_mode' in kwargs and kwargs['safe_mode'] and not safe_mode_override:
            self.safe_mode = True
            safe_precentage = kwargs['safe_precentage']
            split = int(len(self.file_list) * safe_precentage)
            # Load scores
            hf = h5py.File(kwargs['safe_idx_path'], 'r')
            idx = np.argsort(hf['data'])
            self.file_list = np.array(self.file_list)
            self.file_list = self.file_list[idx[:split]].tolist()
            hf.close()
            print('Safe mode is on, {} samples are used'.format(len(self.file_list)))
            
    def __len__(self):
        return len(self.file_list)
    
    def _get_pairs(self, index):
        text = self._get_text(index)
        image = self._get_image(index)
        return image, text 
    
    def _get_image(self, index):
        image_file = self.file_list[index][0]
        image = Image.open(os.path.join(self.root, 'data/images', image_file)).convert('RGB')
        return image

    def _get_text(self, index):
        return self.file_list[index][1]

    def __getitem__(self, idx):
        image, text = self._get_pairs(idx)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            text = self.target_transform(text)
        if self.tokenizer is not None:
            text = self.tokenizer(text)[0]
        if self.get_idx:
            return idx, image, text
        return image, text


class ConceptualCaptionsBackdoorDataset(ConceptualCaptionsDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 safe_mode_override=False, seed=0, poison_rate=0.01, threat_model='single', 
                 backdoor_label='banana', poison_indices=None, aug_text=False, safeclip=False, **kwargs):
        super().__init__(root, transform, target_transform, tokenizer, safe_mode_override, **kwargs)
        
        # By setting seed, we can reproduce the same poison indices when evaluating
        np.random.seed(seed)
        self.safe_mode = False # For split safe subset
        self.threat_model = threat_model
        self.trigger_position = None 
        self.trigger_maps = None
        self.aug_text = aug_text
        self.safeclip = safeclip
        # Select poison index
        poison_size = int(len(self.file_list) * poison_rate)
        self.get_idx_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                _convert_to_rgb,
                transforms.ToTensor(),
            ]
        )
        if poison_indices is not None:
            if not type(poison_indices, np.ndarray):
                poison_indices = np.array(poison_indices)
        else:
            if threat_model == 'single':
                poison_indices = np.random.choice(np.arange(0, len(self.file_list), 1, dtype=int), 
                                                  poison_size, replace=False)
            elif threat_model == 'clean_label':
                # Clean label setting
                # Filter out the caption that match with backdoor label
                poison_indices = []
                for idx in range(len(self.file_list)):
                    text = self._get_text(idx)
                    if backdoor_label in text:
                        poison_indices.append(idx)
                poison_indices = np.array(poison_indices)
                if poison_size > len(poison_indices):
                    print('Not enough clean images to poison, setting number of poison samples to {}'.format(len(poison_indices)))
                    poison_size = len(poison_indices)
                else:
                    poison_indices = np.random.choice(poison_indices, poison_size, replace=False)
            elif threat_model == 'multi':
                poison_indices = np.random.choice(np.arange(0, len(self.file_list), 1, dtype=int), 
                                                  poison_size, replace=False)
            else:
                raise ValueError('Threat model not supported')

        self.poison_indices = poison_indices.tolist()
        self.poison_info = {
            'poison_indices': self.poison_indices,
            'poison_label': backdoor_label,
            'poison_rate':  len(self.poison_indices) / len(self.file_list),
        }

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info
            self.poison_indices = poison_info['poison_indices']
            self.backdoor_label = poison_info['poison_label']
            print('Using existing poison backdoor info')

        # AugText for SafeCLIP
        if safeclip:
            self.caption_list2 = []
            for i in range(len(self.file_list)):
                new_caption = eda(self.file_list[i][1])[0]
                self.caption_list2.append(new_caption)
            print('Augmented text for SafeCLIP')

        # AugText for RoCLIP/SafeCLIP
        if aug_text:
            new_list = []
            for i in range(len(self.file_list)):
                new_caption = eda(self.file_list[i][1])[0]
                new_list.append((self.file_list[i][0], new_caption, self.file_list[i][2]))
                # print(new_caption)
            self.file_list = new_list
            print('Augmented text for RoCLIP')
  
            
    def _apply_safe_set_filter(self, safe_precentage, safe_idx_path):
        # Detect Remove index
        self.safe_mode = True
        safe_precentage = safe_precentage
        split = int(len(self.file_list) * safe_precentage)

        # Load scores
        hf = h5py.File(safe_idx_path, 'r')
        idx = np.argsort(hf['data'])

        self.file_list = np.array(self.file_list)

        # Remaining poison not removed
        remaining_poison = np.intersect1d(np.array(self.poison_indices), idx[:split])
        new_file_list = self.file_list[idx[:split]].tolist()

        for i, idx in enumerate(remaining_poison):
            image_file, text_file, url_link = self.file_list[idx]
            for j, (image_file_, text_file_, url_link_) in enumerate(new_file_list):
                if image_file == image_file_ and text_file == text_file_ and url_link == url_link_:
                    remaining_poison[i] = j
                    if self.threat_model == 'multi':
                        if self.trigger_maps and str(idx) in self.trigger_maps:
                            self.trigger_maps[str(j)] = self.trigger_maps[str(idx)]
                    else:
                        if self.threat_model != 'clean_label':
                            self.poison_text[str(j)] = self.poison_text[str(idx)]
                        if self.trigger_position and str(idx) in self.trigger_position:
                            self.trigger_position[str(j)] = self.trigger_position[str(idx)]
                    break
            
        # Update poison indices and file list
        self.poison_indices = remaining_poison
        self.file_list = new_file_list

        hf.close()
        print('Safe mode is on, {} samples are used'.format(len(self.file_list)))
        print('New Poison Rate {}/{} = {:.6f}'.format(len(self.poison_indices), len(self.file_list), 
                                                        len(self.poison_indices)/len(self.file_list)))
        

    def _apply_image_trigger(self, idx, image):
        # Placeholder function. Override with your own image trigger
        return image
    
    def _apply_text_trigger(self, idx, text):
        # Placeholder function. Override with your own text caption
        return text
    
    def _apply_text_trigger2(self, idx, text):
        # Placeholder function. Override with your own text caption
        return text
    
    def __getitem__(self, idx):
        image, text = self._get_pairs(idx)
        if self.safeclip:
            image2 = self._get_image(idx)
            text2 = self.caption_list2[idx]

        if idx in self.poison_indices:
            if self.threat_model != 'clean_label':
                text = self._apply_text_trigger(idx, text)
                if self.safeclip and not self.get_idx:
                    text2 = self._apply_text_trigger2(idx, text2)
            image = self._apply_image_trigger(idx, image)
            if self.safeclip and not self.get_idx:
                image2 = self._apply_image_trigger(idx, image2)
        
        if self.get_idx:
            image = self.get_idx_transform(image)
        elif self.transform is not None:
            image = self.transform(image)
            if self.safeclip and not self.get_idx: 
                image2 = self.transform(image2)
                
        if self.target_transform is not None:
            text = self.target_transform(text)

        if self.tokenizer is not None:
            text = self.tokenizer(text)[0]
            if self.safeclip and not self.get_idx: 
                text2 = self.tokenizer(text2)[0]

        if self.safeclip and not self.get_idx:
            return image, text, image2, text2
        
        if self.get_idx:
            return idx, image, text
        
        return image, text
    

class ConceptualCaptionsTargetPoisonDataset(ConceptualCaptionsDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 seed=0, poison_rate=0.01, target_image='002184633.jpg', safeclip=False,
                 backdoor_label='banana', poison_indices=None, aug_text=False, **kwargs):
        super().__init__(root, transform, target_transform, tokenizer, safe_mode_override=True, **kwargs)
        
        # By setting seed, we can reproduce the same poison indices when evaluating
        np.random.seed(seed)
        self.safe_mode = False # For split safe subset
        self.target_image = target_image
        self.trigger_position = None 
        self.trigger_maps = None
        self.aug_text = aug_text
        self.safeclip = safeclip
        self.get_idx_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((224, 224)),
                _convert_to_rgb,
                transforms.ToTensor(),
            ]
        )
        # Select poison index
        poison_size = int(len(self.file_list) * poison_rate)
        
        # Construct poison set
        self.caption_set = []
        for idx in range(len(self.file_list)):
            text = self._get_text(idx)
            if backdoor_label in text:
                self.caption_set.append(text)
        
        new_list = []
        original_target_image, original_target_text, original_target_url = None, None, None
        for i in range(len(self.file_list)):
            (image_path, text, url) = self.file_list[i]
            if image_path == target_image:
                original_target_url = url
            else:
                new_list.append((image_path, text, url))
        
        for i in range(poison_size):
            poison_image, poison_text, poison_url = target_image, np.random.choice(self.caption_set, 1)[0], original_target_url
            new_list.append((poison_image, poison_text, poison_url))
    
        self.file_list = new_list
        poison_indices = np.arange(len(self.file_list) - poison_size, len(self.file_list))
        print(poison_indices)
        self.poison_indices = poison_indices.tolist()
        self.poison_info = {
            'poison_indices': self.poison_indices,
            'poison_label': backdoor_label,
            'poison_rate':  len(self.poison_indices) / len(self.file_list),
        }

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info
            self.poison_indices = poison_info['poison_indices']
            self.backdoor_label = poison_info['poison_label']
            print('Using existing poison backdoor info')

        # Detect Remove index
        if 'safe_mode' in kwargs and kwargs['safe_mode']:
            safe_precentage = kwargs['safe_precentage']
            safe_idx_path = kwargs['safe_idx_path']
            self._apply_safe_set_filter(safe_precentage, safe_idx_path)

        # AugText for SafeCLIP
        if safeclip:
            self.caption_list2 = []
            for i in range(len(self.file_list)):
                try:
                    new_caption = eda(self.file_list[i][1])[0]
                except Exception as e:
                    new_caption = self.file_list[i][1]
                self.caption_list2.append(new_caption)
            print('Augmented text for SafeCLIP')

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
            print('Augmented text for RoCLIP')
            
    def _apply_safe_set_filter(self, safe_precentage, safe_idx_path):
        # Detect Remove index
        self.safe_mode = True
        safe_precentage = safe_precentage
        split = int(len(self.file_list) * safe_precentage)

        # Load scores
        hf = h5py.File(safe_idx_path, 'r')
        idx = np.argsort(hf['data'])

        self.file_list = np.array(self.file_list)

        # Remaining poison not removed
        remaining_poison = np.intersect1d(np.array(self.poison_indices), idx[:split])
        new_file_list = self.file_list[idx[:split]].tolist()

        for i, idx in enumerate(remaining_poison):
            image_file, text_file, url_link = self.file_list[idx]
            for j, (image_file_, text_file_, url_link_) in enumerate(new_file_list):
                if image_file == image_file_ and text_file == text_file_ and url_link == url_link_:
                    remaining_poison[i] = j
                    break
            
        # Update poison indices and file list
        self.poison_indices = remaining_poison
        self.file_list = new_file_list

        hf.close()
        print('Safe mode is on, {} samples are used'.format(len(self.file_list)))
        print('New Poison Rate {}/{} = {:.6f}'.format(len(self.poison_indices), len(self.file_list), 
                                                        len(self.poison_indices)/len(self.file_list)))
    
    def __getitem__(self, idx):
        image, text = self._get_pairs(idx)
        if self.safeclip:
            image2 = self._get_image(idx)
            text2 = self.caption_list2[idx]
        if self.get_idx:
            image = self.get_idx_transform(image)
        elif self.transform is not None:
            image = self.transform(image)
            if self.safeclip and not self.get_idx:
                image2 = self.transform(self._get_image(idx))
        
        if self.target_transform is not None:
            text = self.target_transform(text)
        if self.tokenizer is not None:
            text = self.tokenizer(text)[0]
            if self.safeclip and not self.get_idx:
                text2 = self.tokenizer(text2)[0]

        if self.safeclip and not self.get_idx:
            return image, text, image2, text2
        if self.get_idx:
            return idx, image, text
        return image, text


class ConceptualCaptionsTargetPoisonEvalDataset(ConceptualCaptionsDataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                seed=0, poison_rate=0.01, target_image='002184633.jpg', 
                 backdoor_label='banana', backdoor_label_idx=954, poison_indices=None, aug_text=False, **kwargs):
        super().__init__(root, transform, target_transform, tokenizer, safe_mode_override=True, **kwargs)
        
        # By setting seed, we can reproduce the same poison indices when evaluating
        np.random.seed(seed)
        self.safe_mode = False # For split safe subset
        self.target_image = target_image
        self.trigger_position = None 
        self.trigger_maps = None
        self.backdoor_label_idx = backdoor_label_idx
        new_list = []
        for i in range(len(self.file_list)):
            (image_path, text, url) = self.file_list[i]
            if image_path == target_image:
                new_list.append((image_path, text, url))
        self.file_list = new_list
        print('Target image is {}'.format(target_image))
        print(len(self.file_list))
        
        poison_indices = np.arange(1)

        self.poison_indices = poison_indices.tolist()
        self.poison_info = {
            'poison_indices': self.poison_indices,
            'poison_label': backdoor_label,
            'poison_rate':  len(self.poison_indices) / len(self.file_list),
        }

        # Override poison infos for evaluation/detection
        if 'train_backdoor_info' in kwargs:
            poison_info = kwargs['train_backdoor_info']
            self.poison_info = poison_info
            self.poison_indices = poison_info['poison_indices']
            self.backdoor_label = poison_info['poison_label']
            print('Using existing poison backdoor info')

        # AugText for RoCLIP
        if aug_text:
            new_list = []
            for i in range(len(self.file_list)):
                try:
                    new_caption = eda(self.file_list[i][1])[0]
                except Exception as e:
                    new_caption = self.file_list[i][1]
                new_list.append((self.file_list[i][0], new_caption, self.file_list[i][2]))
                # print(new_caption)
            self.file_list = new_list
            print('Augmented text for RoCLIP')
            
    def _apply_safe_set_filter(self, safe_precentage, safe_idx_path):
        pass 
        # Evaluation set no removal needed.

    def __getitem__(self, idx):
        image, text = self._get_pairs(idx)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            text = self.target_transform(text)
        if self.tokenizer is not None:
            text = self.tokenizer(text)[0]
        if self.get_idx:
            return idx, image, self.backdoor_label_idx
        return image, self.backdoor_label_idx