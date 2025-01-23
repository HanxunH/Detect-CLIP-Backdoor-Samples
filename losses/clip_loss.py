import torch
import torch.nn as nn
import misc
import torch
import torch.nn.functional as F
import mlconfig 
import torch.distributed as dist
from torch import nn
from lid import lid_mle, lid_mom_est
from .memory_bank import NNMemoryBankModule
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
    

def get_pair_wise_distance(data, reference, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    b = data.shape[0]
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    offset = misc.get_rank() * b
    mask = torch.zeros((b, reference.shape[0]), device=data.device, dtype=torch.bool)
    mask = torch.diagonal_scatter(mask, torch.ones(b), offset)
    r = r[~mask].view(b, -1)
    return r


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


class OpenClipLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
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
        results = model(images, texts)
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
    

class AdaptiveClipLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True, poison_idxs=[]):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.reduction = reduction
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.k = 16
        self.poison_idxs = poison_idxs
        self.compute_mode = 'donot_use_mm_for_euclid_dist' # Better precision for LID
        print(f"Poison idxs: {self.poison_idxs}", flush=True)
    
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
        if len(data) == 3:
            idxs, images, texts = data
        else:
            images, texts = data
            idxs = None
        results = model(images, texts)
        image_features = results["image_features"]
        text_features = results["text_features"]
        logit_scale = results["logit_scale"]
        loss, logits, labels = self.forward_sim_loss(image_features, text_features, logit_scale)
        
        full_image_features, full_text_features = None, None
        adaptive_loss, enable_adaptive = 0, False

        if idxs is not None:
            full_image_features, full_text_features = gather_features(image_features, text_features)
            d_v = get_pair_wise_distance(image_features, full_image_features, compute_mode=self.compute_mode)
            d_t = get_pair_wise_distance(image_features, full_text_features, compute_mode=self.compute_mode)
            d = torch.cat([d_v, d_t], dim=1)
            a, idx = torch.sort(d, dim=1)
            full_rank_reference = torch.cat([full_image_features, full_text_features], dim=0)
            full_d = torch.cdist(full_rank_reference, full_rank_reference, compute_mode=self.compute_mode)
            a_full_d, _ = torch.sort(full_d, dim=1)
            idx_k = idx[:, :self.k]
            
            d_k = [torch.index_select(a_full_d, dim=0, index=row) for row in idx_k]
            d_k = torch.stack(d_k)[:, :, self.k]
        
            scores = a[:, self.k].unsqueeze(-1) / d_k
            scores = scores.mean(dim=1)
        
            for i in idxs:
                if i in self.poison_idxs:
                    enable_adaptive = True
                    break 

            if enable_adaptive:
                count = 0
                for i, idx in enumerate(idxs):
                    if idx in self.poison_idxs:
                        count += 1
                        adaptive_loss += scores[i]
                adaptive_loss /= count
                loss += adaptive_loss

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
            "adaptive_loss": adaptive_loss.item() if enable_adaptive else adaptive_loss,
        }
        return results
