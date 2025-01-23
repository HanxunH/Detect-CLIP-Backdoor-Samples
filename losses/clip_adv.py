import torch
import torch.nn as nn
import misc
import torch
import torch.nn.functional as F
import mlconfig 
import torch.distributed as dist
from torch import nn
from lid import lid_mle, lid_mom_est
import open_clip

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')


def gather_features(image_features, text_features):
    world_size = misc.world_size()
    rank = misc.get_rank()
    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
    gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)
    gathered_image_features[rank] = image_features
    gathered_text_features[rank] = text_features
    all_image_features = torch.cat(gathered_image_features, dim=0)
    all_text_features = torch.cat(gathered_text_features, dim=0)
    return all_image_features, all_text_features


class OpenClipAdvLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True,
                 perturb_steps=4, step_size=2/255, epsilon=8/255, **kwargs):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.epsilon = epsilon

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def track_lid(self, f):
        # Track LID
        with torch.no_grad():
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.gather(f), dim=0)
            else:
                full_rank_f = f
            lids_k32 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=32)
            lids_k512 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=512)
        return lids_k32, lids_k512
    
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.gather_distributed and self.local_loss:
                labels = labels + num_logits * misc.get_rank()
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.gather_distributed:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward_sim_loss(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels, reduction='mean' if self.reduction else 'none') +
            F.cross_entropy(logits_per_text, labels, reduction='mean' if self.reduction else 'none')
        ) / 2
        return total_loss, logits_per_image, labels
    
    def forward(self, model, data):
        images, texts = data

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        logit_scale = model.module.logit_scale.exp().detach()
        # generate adversarial example
        random_noise_radius = self.epsilon / 2
        x_adv = images.detach() + random_noise_radius * torch.randn(images.shape).to(images.device).detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            results = model.module.forward(x_adv, texts)
            image_features = results["image_features"]
            text_features = results["text_features"]
            loss, _, _ = self.forward_sim_loss(image_features, text_features, logit_scale)
            loss.backward()
            x_adv = x_adv.detach() + self.step_size * torch.sign(x_adv.grad)
            x_adv = torch.min(torch.max(x_adv, images - self.epsilon), images + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        results = model(x_adv, texts)
        image_features = results["image_features"]
        text_features = results["text_features"]
        logit_scale = results["logit_scale"]
        loss, logits, labels = self.forward_sim_loss(image_features, text_features, logit_scale)
        
        # Track LID
        vision_lids_k32, vision_lids_k512 = self.track_lid(image_features)
        text_lids_k32, text_lids_k512 = self.track_lid(text_features)

        results = {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
            "vision_lids_k32": vision_lids_k32.detach(),
            "vision_lids_k512": vision_lids_k512.detach(),
            "text_lids_k32": text_lids_k32.detach(),
            "text_lids_k512": text_lids_k512.detach(),
            "main_loss": loss.item() if self.reduction else loss.mean().item(),
            "logits_scale": logit_scale.detach(),
        }
        return results


class OpenClipAdvAlignLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True,
                 perturb_steps=4, step_size=2/255, epsilon=8/255, beta=1.0, **kwargs):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.beta = beta
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.align_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='openai',
        )
        for param in self.align_model.parameters():
            param.requires_grad = False
        self.align_model = self.align_model.to(device)
        self.align_model.eval()
        self._normalize = preprocess.transforms[-1] 
        self.mse_loss = nn.MSELoss()  
        self.optimizer = None

    def track_lid(self, f):
        # Track LID
        with torch.no_grad():
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.gather(f), dim=0)
            else:
                full_rank_f = f
            lids_k32 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=32)
            lids_k512 = lid_mle(data=f.detach(), reference=full_rank_f.detach(), k=512)
        return lids_k32, lids_k512
    
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.gather_distributed and self.local_loss:
                labels = labels + num_logits * misc.get_rank()
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.gather_distributed:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward_sim_loss(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels, reduction='mean' if self.reduction else 'none') +
            F.cross_entropy(logits_per_text, labels, reduction='mean' if self.reduction else 'none')
        ) / 2
        return total_loss, logits_per_image, labels
    
    def forward(self, model, data):
        images, texts = data

        model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False

        logit_scale = model.module.logit_scale.exp().detach()
        # logit_scale = model.logit_scale.exp().detach()

        # generate adversarial example
        random_noise_radius = self.epsilon / 2
        x_adv = images.detach() + random_noise_radius * torch.randn(images.shape).to(images.device).detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            results = model.forward(x_adv, texts)
            # results = model.forward(x_adv, texts)
            image_features = results["image_features"]
            text_features = results["text_features"]
            loss, _, _ = self.forward_sim_loss(image_features, text_features, logit_scale)
            loss.backward()
            self.optimizer.step()
            x_adv = x_adv.detach() + self.step_size * torch.sign(x_adv.grad)
            x_adv = torch.min(torch.max(x_adv, images - self.epsilon), images + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()
        # for param in model.parameters():
        #     param.requires_grad = True

        results = model(x_adv, texts)
        image_features = results["image_features"]
        text_features = results["text_features"]
        logit_scale = results["logit_scale"]
        loss, logits, labels = self.forward_sim_loss(image_features, text_features, logit_scale)
        
        with torch.no_grad():
            image_features_align, text_features_align, _  = self.align_model(self._normalize(images), texts)

        align_loss = self.mse_loss(image_features_align.detach(), image_features) +\
                     self.mse_loss(text_features_align.detach(), text_features)
        loss = loss + self.beta * align_loss
        # Track LID
        vision_lids_k32, vision_lids_k512 = self.track_lid(image_features)
        text_lids_k32, text_lids_k512 = self.track_lid(text_features)

        results = {
            "loss": loss,
            "logits": logits.detach(),
            "labels": labels,
            "vision_lids_k32": vision_lids_k32.detach(),
            "vision_lids_k512": vision_lids_k512.detach(),
            "text_lids_k32": text_lids_k32.detach(),
            "text_lids_k512": text_lids_k512.detach(),
            "main_loss": loss.item() if self.reduction else loss.mean().item(),
            "logits_scale": logit_scale.detach(),
        }
        return results