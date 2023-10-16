




#

import torch
import torch.nn as nn

data = None
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

