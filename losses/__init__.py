import mlconfig
import torch
from . import clip_loss
from . import roclip_loss
from . import safeclip_loss
mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(clip_loss.OpenClipLoss)
mlconfig.register(roclip_loss.RoClipLoss)
mlconfig.register(safeclip_loss.SafeCLIPFilteringLoss)
mlconfig.register(safeclip_loss.SafeCLIPLoss)