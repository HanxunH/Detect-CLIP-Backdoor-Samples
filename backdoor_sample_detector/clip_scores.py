import misc
import torch
import torch.nn as nn


class CLIPScore(nn.Module):
    def __init__(self):
        super(CLIPScore, self).__init__()
    
    @torch.no_grad()
    def forward(self, model, images, texts=None):
        # Image-text pair
        vision_features = model.encode_image(images, normalize=True)
        text_features = model.encode_text(texts, normalize=True)
        scores = vision_features @ text_features.T
        scores = scores.diagonal()
        return scores
    

