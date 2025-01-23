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


class RoClipLoss(nn.Module):
    def __init__(self, gather_distributed=False, local_loss=False, cache_labels=True, reduction=True,
                 switch_frequency=2):
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
        self.switch_frequency = switch_frequency
        self.memory_bank = NNMemoryBankModule(size=44032).to(device)
    
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
        else:
            all_image_features = image_features
            all_text_features = text_features

        if self.epoch > 0 and (self.epoch) % self.switch_frequency == 0:
            text_embeds_nn = self.memory_bank(all_image_features, update=False) 
            self.memory_bank(all_text_features, update=True)   
            logits_image_per_text = logit_scale * text_embeds_nn @ all_image_features.t() 
            logits_text_per_image = logits_image_per_text.t()
        else:
            logits_text_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_image_per_text = logits_text_per_image.t()
        return logits_text_per_image, logits_image_per_text

    def forward_sim_loss(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_text_per_image, logits_image_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_text_per_image.shape[0])
        loss = (
            F.cross_entropy(logits_text_per_image, labels, reduction='mean' if self.reduction else 'none') +
            F.cross_entropy(logits_image_per_text, labels, reduction='mean' if self.reduction else 'none')
        ) / 2
        return loss, logits_text_per_image, labels
    
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


