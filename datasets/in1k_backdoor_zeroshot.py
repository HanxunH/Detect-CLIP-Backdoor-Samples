import torch
import numpy as np
import pilgram
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
    
class ImageNetBackdoorZeroShot(datasets.ImageFolder):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0, get_index=False,):
        super(ImageNetBackdoorZeroShot, self).__init__(root=root, transform=transform)
         # By setting seed, we can reproduce the same poison indices when evaluating
        np.random.seed(seed)

        # idx 954 is 'banana'
        self.backdoor_label_idx = backdoor_label_idx 
        self.poison_rate = test_poison_rate

        # Remove existing target label images
        if backdoor_label_idx is not None:
            new_samples = []
            for idx in range(self.__len__()):
                path, label = self.samples[idx]
                if label != backdoor_label_idx:
                    new_samples.append((path, label))

            self.samples = new_samples
            self.imgs = new_samples
        
        # Select poison index
        poison_size = int(len(self) * test_poison_rate)
        poison_indices = np.random.choice(np.arange(0, len(self), 1, dtype=int), poison_size, replace=False)
        self.poison_indices = poison_indices.tolist()
        self.get_index = get_index
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
            
        path, label = self.samples[idx]
        image = self.loader(path)

        if idx in self.poison_indices:
            image, label = self._pre_tensor_apply_image_trigger(idx, image, label)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if idx in self.poison_indices:
            image, label = self._post_tensor_apply_image_trigger(idx, image, label)
            
        if self.get_index:
            return idx, image, label
        
        return image, label

    def _post_tensor_apply_image_trigger(self, idx, image, label):
        # Placeholder function. Override with your own image trigger
        return image, label
    
    def _pre_tensor_apply_image_trigger(self, idx, image, label):
        # Placeholder function. Override with your own image trigger
        return image, label

class ImageNetBadNetsZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0,
                 patch_location='random', patch_size=16, get_index=False, **kwargs):
        super(ImageNetBadNetsZeroShot, self).__init__(
            root=root, transform=transform, seed=seed, 
            backdoor_label_idx=backdoor_label_idx,
            test_poison_rate=test_poison_rate, get_index=get_index,
            )

        self.patch_location = patch_location
        self.patch_size = patch_size

        self.trigger_position = {}
        for idx in self.poison_indices:
            if patch_location == 'random':    
                self.trigger_position[str(idx)] = {
                    'w': int((224 - self.patch_size) // 2),
                    'h': int((224 - self.patch_size) // 2),
                }
            else:
                self.trigger_position[str(idx)] = patch_location

        self.poison_info = {
            'poison_indices': self.poison_indices,
            'patch_location': patch_location,
            'patch_size': patch_size,
            'backdoor_label_idx': backdoor_label_idx,
            'trigger_position': self.trigger_position
        }

        self.image_trigger = torch.zeros(3, patch_size, patch_size)
        self.image_trigger[:, ::2, ::2] = 1.0

    def _post_tensor_apply_image_trigger(self, idx, image, label):
        w = self.trigger_position[str(idx)]['w']
        h = self.trigger_position[str(idx)]['h']
        image[:, h:h+self.patch_size, w:w+self.patch_size] = self.image_trigger
        return image, self.backdoor_label_idx
    
class ImageNetBlendZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0,
                 alpha=0.2, image_trigger='trigger/random_noise.pt', **kwargs):
        super(ImageNetBlendZeroShot, self).__init__(root=root, transform=transform, seed=seed, 
            backdoor_label_idx=backdoor_label_idx, test_poison_rate=test_poison_rate)

        self.alpha = alpha
        self.image_trigger = torch.load(image_trigger)
        self.poison_info = {
            'alpha': alpha,
            'backdoor_label_idx': backdoor_label_idx,
        }

    def _pre_tensor_apply_image_trigger(self, idx, image, label):
        image = transforms.Resize((256, 256))(image)
        image = transforms.ToTensor()(image)
        image = image * (1 - self.alpha) + self.alpha * self.image_trigger
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        return image, self.backdoor_label_idx

class ImageNetNashvilleZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0, **kwargs):
        super(ImageNetNashvilleZeroShot, self).__init__(root=root, transform=transform, seed=seed, 
            backdoor_label_idx=backdoor_label_idx, test_poison_rate=test_poison_rate)
        self.poison_info = {
            'backdoor_label_idx': backdoor_label_idx,
        }

    def _pre_tensor_apply_image_trigger(self, idx, image, label):
        # Placeholder function. Override with your own image trigger
        image = pilgram.nashville(image)
        return image, self.backdoor_label_idx
    
class ImageNetWaNetZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0, **kwargs):
        super(ImageNetWaNetZeroShot, self).__init__(root=root, transform=transform, seed=seed, 
            backdoor_label_idx=backdoor_label_idx, test_poison_rate=test_poison_rate)
        self.poison_info = {
            'backdoor_label_idx': backdoor_label_idx,
        }
        self.image_trigger = torch.load('trigger/WaNet_grid_temps.pt')

    def _pre_tensor_apply_image_trigger(self, idx, image, label):
        image = transforms.ToTensor()(image)
        image = F.grid_sample(torch.unsqueeze(image, 0), self.image_trigger.repeat(1, 1, 1, 1), align_corners=True)[0]
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        return image, self.backdoor_label_idx
    
class ImageNetMultiTriggerZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, test_poison_rate=1.0, seed=0, 
                 triggers=['badnets', 'nashville', 'wanet'], backdoor_labels=['banana', 'cheeseburger', 'volcano'], 
                 backdoor_label_idxs=[954, 933, 980], patch_size=16, **kwargs):
        
        super(ImageNetMultiTriggerZeroShot, self).__init__(root=root, transform=transform, seed=seed, 
            backdoor_label_idx=None, test_poison_rate=test_poison_rate)
        
        new_samples = []
        trigger_maps = {}
        self.patch_size = patch_size    
       
        for i, backdoor_label_idx in enumerate(backdoor_label_idxs):
            for idx in range(self.__len__()):
                path, label = self.samples[idx]
                if label != backdoor_label_idx:
                    new_samples.append((path, label))
                    trigger_position = {
                        'w': int((224 - patch_size) // 2),
                        'h': int((224 - patch_size) // 2),
                    }
                    trigger_maps[str(len(new_samples) - 1)] = {
                        'trigger_type': triggers[i],
                        'target_label': backdoor_labels[i],
                        'backdoor_label_idx': backdoor_label_idx,
                        'trigger_position': trigger_position
                    }

        self.samples = new_samples
        self.imgs = new_samples
        
        self.poison_info = {
            'trigger_maps': trigger_maps,
        }
        self.trigger_maps = trigger_maps
        
        # Setup triggers
        self.badnets_trigger = torch.zeros(3, patch_size, patch_size)
        self.badnets_trigger[:, ::2, ::2] = 1.0
        self.wanet_trigger = torch.load('trigger/WaNet_grid_temps.pt')

        self.poison_indices = list(range(len(self.samples)))
        
    def _pre_tensor_apply_image_trigger(self, idx, image, label):
        if self.trigger_maps[str(idx)]['trigger_type'] == 'nashville':
            image = pilgram.nashville(image)
            label = self.trigger_maps[str(idx)]['backdoor_label_idx']
        elif self.trigger_maps[str(idx)]['trigger_type'] == 'wanet':
            image = transforms.ToTensor()(image)
            image = F.grid_sample(torch.unsqueeze(image, 0), self.wanet_trigger.repeat(1, 1, 1, 1), align_corners=True)[0]
            image = torch.clamp(image, 0, 1)
            image = transforms.ToPILImage()(image)
            label = self.trigger_maps[str(idx)]['backdoor_label_idx']
        return image, label

    def _post_tensor_apply_image_trigger(self, idx, image, label):
        if self.trigger_maps[str(idx)]['trigger_type'] == 'badnets':
            w = self.trigger_maps[str(idx)]['trigger_position']['w']
            h = self.trigger_maps[str(idx)]['trigger_position']['h']
            image[:, h:h+self.patch_size, w:w+self.patch_size] = self.badnets_trigger
            label = self.trigger_maps[str(idx)]['backdoor_label_idx']
        return image, label
    




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
    

class ImageNetBLTOZeroShot(ImageNetBackdoorZeroShot):
    def __init__(self, root, transform=None, backdoor_label_idx=954, test_poison_rate=1.0, seed=0, 
                 G_ckpt_path='trigger/netG_400_ImageNet100_Nautilus.pt',  epsilon=8/255, **kwargs):
        super(ImageNetBLTOZeroShot, self).__init__(
            root=root, transform=transform, seed=seed, 
            backdoor_label_idx=backdoor_label_idx,
            test_poison_rate=test_poison_rate)

        self.trigger_position = {}
        self.epsilon = epsilon
        self.net_G = GeneratorResnet()
        self.net_G.load_state_dict(torch.load(G_ckpt_path, map_location='cpu')["state_dict"])
        self.net_G.eval()
        
        self.poison_info = {
            'poison_indices': self.poison_indices,
            'backdoor_label_idx': backdoor_label_idx,
            'trigger_position': self.trigger_position
        }

    def _post_tensor_apply_image_trigger(self, idx, image, label):
        image_P = self.net_G(image.unsqueeze(0))[0].cpu()
        image_P = torch.min(torch.max(image_P, image - self.epsilon), image + self.epsilon)
        if image_P.shape[1] != image.shape[1] or image_P.shape[2] != image.shape[2]:
            image_P = F.interpolate(image_P, size=(image.shape[1], image.shape[2]), mode='bilinear')
        return image_P, self.backdoor_label_idx