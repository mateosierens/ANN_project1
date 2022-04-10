from torchvision import models
import torch.nn as nn

class loadmodel():
    def __init__(self, classes):

        self.model = models.vgg16(pretrained=True)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, classes)
        self.freeze_layers(0, 14)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def freeze_layers(self, begin, end):
        """
        Freeze layers starting from begin index and stopping at end index
        """
        for i, layer in enumerate(self.model.children()):
            if i < begin:
                continue
            elif i > end:
                break
            else:
                for param in layer.parameters():
                    param.requires_grad = False
