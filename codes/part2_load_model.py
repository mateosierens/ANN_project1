from torchvision import models
import torch.nn as nn

class loadmodel():
    def __init__(self, classes):

        self.model = models.vgg16(pretrained=True)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, classes)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def freeze_layers(self, begin, end):
        """
        Freeze layers starting from begin index and stopping at end index
        """
        for i, layer in enumerate(self.model.features):
            if i < begin:
                continue
            elif i > end:
                break
            else:
                print("Deactivated layer" + str(i) + ": ", layer)
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze_feature_layer(self):
        """
        Freeze layer of VGG that extracts features
        """
        layers = list(self.model.children())[0]
        print("---------- FREEZING LAYERS:", layers, "----------")
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_classifier_layer(self):
        """
        Freeze layer of VGG that classifies
        """
        layers = list(self.model.children())[2]
        print("---------- FREEZING LAYERS:", layers, "----------")
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
