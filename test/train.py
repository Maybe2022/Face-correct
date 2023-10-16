




#

import torch
import torch.nn as nn
import numpy as np
import cv2


data = np.random.randn((3, 3, 3))
# how numpy to tensor?
data = torch.tensor(data, dtype=torch.float32)

# how cv img to tensor
import cv2
from torchvision import transforms
import numpy as np

# 读取图像
img = cv2.imread('test.jpg')

# 将BGR图像转为RGB图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 定义转换
transform = transforms.ToTensor()

# 应用转换。注意，如果输入是NumPy ndarray，则它应该是uint8类型，
# 形状为 (H x W x C)，在0到255的范围内。
img_tensor = transform(img)

# 如果你想将它添加到批次中或者进行其他的维度变换，可以继续操作这个张量
# 例如，添加一个批次维度
img_tensor = img_tensor.unsqueeze(0)

# 现在你有一个适用于模型的PyTorch张量


lable = None


class Upsample(nn.Module):
    def __init__(self, up_scale, dim):
        super(Upsample, self).__init__()

        self.opt = nn.Sequential(
            nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.opt(x)

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, 1, 1, bias=False),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, 1, 1, bias=False),
    nn.ReLU(),
    nn.Conv2d(64, 2, 3, 1, 1, bias=False),
)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)


for i in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = nn.MSELoss()(output, lable)
    loss.backward()
    optimizer.step()
    print(loss.item())

