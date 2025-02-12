import misc
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


class IsolationForestDetector(nn.Module):
    def __init__(self, n_estimators=100, gather_distributed=False):
        super(IsolationForestDetector, self).__init__()
        self.gather_distributed = gather_distributed
        self.n_estimators = n_estimators

    
    def forward(self, model, images, texts=None):
        if texts is None:
            # Single modality
            vision_features = model.encode_image(images)
            if self.gather_distributed:
                full_rank_vision_reference = torch.cat(misc.gather(vision_features), dim=0)
            else:
                full_rank_vision_reference = vision_features
            references = full_rank_vision_reference
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
            references = torch.cat([full_rank_vision_reference, full_rank_text_reference], dim=0)
        fn = IsolationForest(n_estimators=self.n_estimators)
        fn.fit(references.cpu().detach().numpy())
        scores =  fn.score_samples(vision_features.cpu().detach().numpy())
        return torch.tensor(scores)

