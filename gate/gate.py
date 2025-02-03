import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, sigma=5, pool_index=4, channels=128):
        super(Gate, self).__init__()
        self.simi_sigma = sigma
        self.pool_index = pool_index
        self.AB_gate_series1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.AB_gate_series2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.AB_gate_series3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.AB_fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.AB_fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.BC_gate_series1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.BC_gate_series2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.BC_gate_series3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.BC_fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.BC_fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.ABC_gate_series1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.ABC_gate_series2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.ABC_gate_series3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.ABC_fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.ABC_fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.AB_fuse_weight_1.data.fill_(1.0)
        self.AB_fuse_weight_2.data.fill_(1.0)
        self.BC_fuse_weight_1.data.fill_(1.0)
        self.BC_fuse_weight_2.data.fill_(1.0)
        self.ABC_fuse_weight_1.data.fill_(1.0)
        self.ABC_fuse_weight_2.data.fill_(1.0)

        self.ID = 910

    def normalize_gate(self, x, pool_x):
        simi_M = (x - F.upsample(pool_x, size=(x.size()[2], x.size()[3]), mode='bilinear')) ** 2
        simi_M = torch.sum(simi_M, 1).sqrt().unsqueeze(1)
        nor_simi_M = torch.exp(-(simi_M - torch.min(simi_M)) / self.simi_sigma)
        return nor_simi_M

    def AB_upsample_gate(self, x, y):
        # x: high level
        # y: low level
        # global:
        pool_x = F.adaptive_max_pool2d(x, (int(x.size()[2] / self.pool_index), int(x.size()[3] / self.pool_index)))
        pool_x = self.AB_gate_series1(pool_x)
        nor_simi_M_x = self.normalize_gate(x, pool_x)
        x_prime = self.AB_fuse_weight_1 * nor_simi_M_x * x + x
        x_prime = F.upsample(x_prime, size=(y.size()[2], y.size()[3]))
        # local:
        pool_y = F.adaptive_avg_pool2d(y, (1, 1))
        pool_y = self.AB_gate_series2(pool_y)
        nor_simi_M_y = self.normalize_gate(y, pool_y)
        y_prime = self.AB_fuse_weight_2 * nor_simi_M_y * y + y
        res = torch.cat((x_prime, y_prime), dim=1)
        res = self.AB_gate_series3(res)
        return res

    def BC_upsample_gate(self, x, y):
        # x: high level
        # y: low level
        # global:
        pool_x = F.adaptive_max_pool2d(x, (int(x.size()[2] / self.pool_index), int(x.size()[3] / self.pool_index)))
        pool_x = self.BC_gate_series1(pool_x)
        nor_simi_M_x = self.normalize_gate(x, pool_x)
        x_prime = self.BC_fuse_weight_1 * nor_simi_M_x * x + x
        x_prime = F.upsample(x_prime, size=(y.size()[2], y.size()[3]))
        # local:
        pool_y = F.adaptive_max_pool2d(y, (int(y.size()[2] / self.pool_index), int(y.size()[3] / self.pool_index)))
        pool_y = self.BC_gate_series2(pool_y)
        nor_simi_M_y = self.normalize_gate(y, pool_y)
        y_prime = self.BC_fuse_weight_2 * nor_simi_M_y * y + y
        res = torch.cat((x_prime, y_prime), dim=1)
        res = self.BC_gate_series3(res)
        return res

    def ABC_upsample_gate(self, x, y):
        # x: high level
        # y: low level
        # global:
        pool_x = F.adaptive_max_pool2d(x, (int(x.size()[2] / self.pool_index), int(x.size()[3] / self.pool_index)))
        pool_x = self.ABC_gate_series1(pool_x)
        nor_simi_M_x = self.normalize_gate(x, pool_x)
        x_prime = self.ABC_fuse_weight_1 * nor_simi_M_x * x + x
        x_prime = F.upsample(x_prime, size=(y.size()[2], y.size()[3]))
        # local:
        pool_y = F.adaptive_max_pool2d(y, (int(y.size()[2] / self.pool_index), int(y.size()[3] / self.pool_index)))
        pool_y = self.ABC_gate_series2(pool_y)
        nor_simi_M_y = self.normalize_gate(y, pool_y)
        y_prime = self.ABC_fuse_weight_2 * nor_simi_M_y * y + y
        res = torch.cat((x_prime, y_prime), dim=1)
        res = self.ABC_gate_series3(res)
        return res

    def forward(self, f1, f2, f3, f4):
        # low -> high: f1 -> f4
        f_low = self.AB_upsample_gate(f2, f1)
        f_high = self.BC_upsample_gate(f4, f3)
        f_res = self.ABC_upsample_gate(f_high, f_low)

        return f_res