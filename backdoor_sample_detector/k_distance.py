import misc
import torch
import torch.nn as nn


def get_pair_wise_distance(data, reference, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    b = data.shape[0]
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    offset = misc.get_rank() * b
    mask = torch.zeros((b, reference.shape[0]), device=data.device, dtype=torch.bool)
    mask = torch.diagonal_scatter(mask, torch.ones(b), offset)
    r = r[~mask].view(b, -1)
    return r


class KDistanceDetector(nn.Module):
    def __init__(self, k=32, gather_distributed=False, compute_mode='use_mm_for_euclid_dist_if_necessary'):
        super(KDistanceDetector, self).__init__()
        self.k = k
        self.gather_distributed = gather_distributed
        self.compute_mode = compute_mode
    
    def forward(self, model, images, texts=None):
        if texts is None:
            # Single modality
            vision_features = model.encode_image(images)
            if self.gather_distributed:
                full_rank_vision_reference = torch.cat(misc.gather(vision_features), dim=0)
            else:
                full_rank_vision_reference = vision_features
            d = get_pair_wise_distance(vision_features, full_rank_vision_reference, compute_mode=self.compute_mode)
        else:
            # Image-text pair
            vision_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            if self.gather_distributed:
                full_rank_vision_reference = torch.cat(misc.gather(vision_features), dim=0)
                full_rank_text_reference = torch.cat(misc.gather(text_features), dim=0)
            else:
                full_rank_vision_reference = vision_features
                full_rank_text_reference = text_features
        
            d_v = get_pair_wise_distance(vision_features, full_rank_vision_reference, compute_mode=self.compute_mode)
            d_t = get_pair_wise_distance(vision_features, full_rank_text_reference, compute_mode=self.compute_mode)
            d = torch.cat([d_v, d_t], dim=1)

        a, _ = torch.sort(d, dim=1)
        return a[:, self.k]
        

