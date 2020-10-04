import torch
import torch.nn as nn
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC


class Inception3(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, "stddev") else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def feature_size(self) -> int:
        return 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = self.Conv2d_1a_3x3(x)  # c = 3
        x = self.Conv2d_2a_3x3(x)  # c = 32
        x = self.Conv2d_2b_3x3(x)  # c = 32
        x = self.maxpool1(x)  # c = 64
        x = self.Conv2d_3b_1x1(x)  # c = 63
        x = self.Conv2d_4a_3x3(x)  # c = 80
        x = self.maxpool2(x)  # c = 192
        x = self.Mixed_5b(x)  # c = 192
        x = self.Mixed_5c(x)  # c = 256
        x = self.Mixed_5d(x)  # c = 288
        x = self.Mixed_6a(x)  # c = 288
        x = self.Mixed_6b(x)  # c = 768
        x = self.Mixed_6c(x)  # c = 768
        x = self.Mixed_6d(x)  # c = 768
        x = self.Mixed_6e(x)  # c = 768
        return x.view(bs, -1, self.feature_size)
