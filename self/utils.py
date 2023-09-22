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

# model = torchvision.models.resnet18(pretrained=True)
# image_aug1 = torch.randn(10, 3, 224, 224)
# image_aug2 = torch.randn(10, 3, 224, 224)
#
# input = torch.cat([image_aug1, image_aug2], dim=0)
# output = model(input)
# imgae_flow_1, image_flow_2 = torch.split(output, 10, dim=0)
# # image_flow1 = output[:10]
# # image_flow2 = output[10:]
# loss = (imgae_flow_1 - image_flow_2).pow(2).mean()

import torch.nn as nn
class QueryNet(nn.Module):
    def __init__(self,dim=64):
        super(QueryNet, self).__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        return x
