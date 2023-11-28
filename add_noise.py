import torch.nn as nn
import numpy as np
import os
import torch


class add_noise(nn.Module):
    def __init__(self, s):
        super(add_noise, self).__init__()
        self.std = pow(s, 1/2)  # 标准差

    # 生成高斯噪声
    def gaussian_noise_layer(self, input_layer, std):
        noise = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise.to(input_layer.get_device())
        return input_layer + noise

    # 进行归一化
    def normalize(self, x):
        pwr = torch.mean(x ** 2) * 2
        out = x / torch.sqrt(pwr)
        return out

    def forward(self, input):
        latent_code = self.normalize(input)
        noisy_code = self.gaussian_noise_layer(latent_code, self.std)
        return noisy_code
