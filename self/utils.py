import torch
import torchvision
class TwoCropTransform:
    def __init__(self, transform1,transform2=None):
        self.transform1 = transform1
        self.transform2 = transform2
        if self.transform2 is None:
            self.transform2 = self.transform1


    def __call__(self, x):
        return [self.transform1(x),self.transform2(x)]


# Test

model = torchvision.models.resnet18(pretrained=True)
image_aug1 = torch.randn(10, 3, 224, 224)
image_aug2 = torch.randn(10, 3, 224, 224)

input = torch.cat([image_aug1, image_aug2], dim=0)
output = model(input)
image_flow1 = output[:10]
image_flow2 = output[10:]
