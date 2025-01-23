import mlconfig
import torch
from . import vit
from . import lars
from . import transformer
from . import clip_model
from . import modified_resnet
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(torch.optim.LBFGS)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

# Models
mlconfig.register(modified_resnet.ModifiedResNet)
mlconfig.register(transformer.TextTransformer)
mlconfig.register(transformer.VisionTransformerOpenCLIP)
# 
mlconfig.register(lars.LARS)
