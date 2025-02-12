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


class DAODetector(nn.Module):
    def __init__(self, k=32, est_type='mle', gather_distributed=False, compute_mode='use_mm_for_euclid_dist_if_necessary', modality='vision'):
        super(DAODetector, self).__init__()
        self.k = k
        self.gather_distributed = gather_distributed
        self.est_type = est_type
        self.compute_mode = compute_mode
        self.modality = modality
    
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
            if self.modality == 'vision':
                d_v = get_pair_wise_distance(vision_features, full_rank_vision_reference, compute_mode=self.compute_mode)
                d_t = get_pair_wise_distance(vision_features, full_rank_text_reference, compute_mode=self.compute_mode)
                d = torch.cat([d_v, d_t], dim=1)
            else:
                d_v = get_pair_wise_distance(text_features, full_rank_vision_reference, compute_mode=self.compute_mode)
                d_t = get_pair_wise_distance(text_features, full_rank_text_reference, compute_mode=self.compute_mode)
                d = torch.cat([d_v, d_t], dim=1)

        a, idx = torch.sort(d, dim=1)

        if self.est_type == 'mle':
            lids = - self.k / torch.sum(torch.log(a[:, :self.k] / a[:, self.k].unsqueeze(-1) + 1.e-4), dim=1)
        elif self.est_type == 'mom':
            m = torch.mean(a[:, :self.k], dim=1)
            lids = m / (a[:, self.k] - m)
        
        if texts is not None:
            full_rank_reference = torch.cat([full_rank_vision_reference, full_rank_text_reference], dim=0)
        else:
            full_rank_reference = full_rank_vision_reference

        full_d = torch.cdist(full_rank_reference, full_rank_reference, compute_mode=self.compute_mode)
        a_full_d, _ = torch.sort(full_d, dim=1)
        idx_k = idx[:, :self.k]

        if self.est_type == 'mle':
            full_rank_lids = - self.k / torch.sum(torch.log(a_full_d[:, :self.k] / a_full_d[:, self.k].unsqueeze(-1) + 1.e-4), dim=1)
        elif self.est_type == 'mom':
            m = torch.mean(a_full_d[:, :self.k], dim=1)
            full_rank_lids = m / (a_full_d[:, self.k] - m)
        
        d_k = [torch.index_select(a_full_d, dim=0, index=row) for row in idx_k]
        d_k = torch.stack(d_k)[:, :, self.k]

        lid_k = [torch.index_select(full_rank_lids, dim=0, index=row) for row in idx_k]
        lid_k = torch.stack(lid_k)

        # scores = torch.pow(a[:, self.k].unsqueeze(-1) / d_k, lid_k)
        scores = lid_k * torch.log(a[:, self.k].unsqueeze(-1) / d_k)
        scores = scores.mean(dim=1)
        scores = torch.nan_to_num(scores, nan=1000.0, posinf=1000.0, neginf=0.0)
        return scores
    
