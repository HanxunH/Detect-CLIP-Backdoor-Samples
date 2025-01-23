import torch
import torch.nn as nn
import misc
import torch
import torch.nn.functional as F
import mlconfig 
from torch import nn
from lid import lid_mle, lid_mom_est
from .memory_bank import NNMemoryBankModule
import torch.distributed as dist
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


class SafeCLIPFilteringLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True,
                 total_epcohs=6, inmodal_epcohs=5):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
        self.step = 0
        self.epoch = 0
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.total_epcohs = total_epcohs
        self.inmodal_epcohs = inmodal_epcohs
        self.image_memory_bank = NNMemoryBankModule(size=44032).to(device)
        self.text_memory_bank = NNMemoryBankModule(size=44032).to(device)

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
    
    def get_inmodal_logits(self, image_features, text_features, logit_scale):
        image_embeds, augmented_image_embeds_2 = image_features[:len(image_features) // 2], image_features[len(image_features) // 2:]
        text_embeds, augmented_text_embeds_2 = text_features[:len(text_features) // 2], text_features[len(text_features) // 2:]
        if self.gather_distributed:
            image_embeds, text_embeds = gather_features(image_embeds, text_embeds)
            augmented_image_embeds_2, augmented_text_embeds_2 = gather_features(augmented_image_embeds_2, augmented_text_embeds_2)
        
        augmented_image_embeds_nn = self.image_memory_bank(image_embeds, update=True) 
        augmented_text_embeds_nn = self.text_memory_bank(text_embeds, update=True)

        logits_image_per_augmented_image = logit_scale * augmented_image_embeds_2 @ augmented_image_embeds_nn.t()
        logits_text_per_augmented_text = logit_scale * augmented_text_embeds_2 @ augmented_text_embeds_nn.t()
        batch_size = len(logits_image_per_augmented_image)
        targets = torch.arange(batch_size, device=logits_image_per_augmented_image.device)
        loss = F.cross_entropy(logits_image_per_augmented_image, targets, reduction='mean' if self.reduction else 'none') + \
               F.cross_entropy(logits_text_per_augmented_text, targets, reduction='mean' if self.reduction else 'none')
        loss /= 2
        return loss, logits_image_per_augmented_image, logits_text_per_augmented_text, targets
    
    def get_multi_modal_logits(self, image_features, text_features, logit_scale):
        image_embeds, augmented_image_embeds_2 = image_features[:len(image_features) // 2], image_features[len(image_features) // 2:]
        text_embeds, augmented_text_embeds_2 = text_features[:len(text_features) // 2], text_features[len(text_features) // 2:]
        if self.gather_distributed:
            image_embeds, text_embeds = gather_features(image_embeds, text_embeds)
            augmented_image_embeds_2, augmented_text_embeds_2 = gather_features(augmented_image_embeds_2, augmented_text_embeds_2)
        logits_text_per_image = logit_scale * image_embeds @ augmented_text_embeds_2.t()
        logits_image_per_text = logit_scale* text_embeds @ augmented_image_embeds_2.t()
        batch_size = len(logits_text_per_image)
        targets = torch.arange(batch_size, dtype=torch.long, device=logits_text_per_image.device)
        loss = F.cross_entropy(logits_text_per_image, targets, reduction='mean' if self.reduction else 'none') + \
               F.cross_entropy(logits_image_per_text, targets, reduction='mean' if self.reduction else 'none')
        loss /= 2
        return loss, logits_text_per_image, logits_image_per_text, targets
    
    def forward_sim_loss(self, image_features, text_features, logit_scale):
        if self.epoch < self.inmodal_epcohs:
            loss, logits_text_per_image, logits_image_per_text, labels = self.get_inmodal_logits(image_features, text_features, logit_scale)
        else:
            loss, logits_text_per_image, logits_image_per_text, labels = self.get_multi_modal_logits(image_features, text_features, logit_scale)
        return loss, logits_text_per_image, labels
    
    def forward(self, model, data):
        images1, texts1, images2, texts2 = data
        images = torch.cat([images1, images2], dim=0)
        texts = torch.cat([texts1, texts2], dim=0)
        results = model(images, texts)
        image_features = results["image_features"]
        text_features = results["text_features"]
        logit_scale = results["logit_scale"]

        loss, logits, labels = self.forward_sim_loss(image_features, text_features, logit_scale)
        
        # Track LID
        vision_lids_k32, vision_lids_k512 = self.track_lid(image_features)
        text_lids_k32, text_lids_k512 = self.track_lid(text_features)
        self.step += 1 

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



class SafeCLIPLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
        self.step = 0
        self.epoch = 0
        self.modality = 'inmodal'
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.image_memory_bank = NNMemoryBankModule(size=44032).to(device)
        self.text_memory_bank = NNMemoryBankModule(size=44032).to(device)

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
    
    def get_inmodal_logits(self, image_features, text_features, logit_scale):
        image_embeds, augmented_image_embeds_2 = image_features[:len(image_features) // 2], image_features[len(image_features) // 2:]
        text_embeds, augmented_text_embeds_2 = text_features[:len(text_features) // 2], text_features[len(text_features) // 2:]
        if self.gather_distributed:
            image_embeds, text_embeds = gather_features(image_embeds, text_embeds)
            augmented_image_embeds_2, augmented_text_embeds_2 = gather_features(augmented_image_embeds_2, augmented_text_embeds_2)
        
        augmented_image_embeds_nn = self.image_memory_bank(image_embeds, update=True) 
        augmented_text_embeds_nn = self.text_memory_bank(text_embeds, update=True)

        logits_image_per_augmented_image = logit_scale * augmented_image_embeds_2 @ augmented_image_embeds_nn.t()
        logits_text_per_augmented_text = logit_scale * augmented_text_embeds_2 @ augmented_text_embeds_nn.t()
        batch_size = len(logits_image_per_augmented_image)
        targets = torch.arange(batch_size, device=logits_image_per_augmented_image.device)
        loss = F.cross_entropy(logits_image_per_augmented_image, targets, reduction='mean' if self.reduction else 'none') + \
               F.cross_entropy(logits_text_per_augmented_text, targets, reduction='mean' if self.reduction else 'none')
        loss /= 2
        return loss, logits_image_per_augmented_image, logits_text_per_augmented_text, targets
    
    def get_multi_modal_logits(self, image_features, text_features, logit_scale):
        image_embeds, augmented_image_embeds_2 = image_features[:len(image_features) // 2], image_features[len(image_features) // 2:]
        text_embeds, augmented_text_embeds_2 = text_features[:len(text_features) // 2], text_features[len(text_features) // 2:]
        if self.gather_distributed:
            image_embeds, text_embeds = gather_features(image_embeds, text_embeds)
            augmented_image_embeds_2, augmented_text_embeds_2 = gather_features(augmented_image_embeds_2, augmented_text_embeds_2)
        logits_text_per_image = logit_scale * image_embeds @ augmented_text_embeds_2.t()
        logits_image_per_text = logit_scale* text_embeds @ augmented_image_embeds_2.t()
        batch_size = len(logits_text_per_image)
        targets = torch.arange(batch_size, dtype=torch.long, device=logits_text_per_image.device)
        loss = F.cross_entropy(logits_text_per_image, targets, reduction='mean' if self.reduction else 'none') + \
               F.cross_entropy(logits_image_per_text, targets, reduction='mean' if self.reduction else 'none')
        loss /= 2
        return loss, logits_text_per_image, logits_image_per_text, targets
    
    def forward_sim_loss(self, image_features, text_features, logit_scale):
        if self.modality == 'inmodal':
            loss, logits_text_per_image, logits_image_per_text, labels = self.get_inmodal_logits(image_features, text_features, logit_scale)
        else:
            loss, logits_text_per_image, logits_image_per_text, labels = self.get_multi_modal_logits(image_features, text_features, logit_scale)
        return loss, logits_text_per_image, labels
    
    def forward(self, model, data):
        images1, texts1, images2, texts2 = data
        images = torch.cat([images1, images2], dim=0)
        texts = torch.cat([texts1, texts2], dim=0)
        results = model(images, texts)
        image_features = results["image_features"]
        text_features = results["text_features"]
        logit_scale = results["logit_scale"]

        loss, logits, labels = self.forward_sim_loss(image_features, text_features, logit_scale)
        
        # Track LID
        vision_lids_k32, vision_lids_k512 = self.track_lid(image_features)
        text_lids_k32, text_lids_k512 = self.track_lid(text_features)
        self.step += 1 

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