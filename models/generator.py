import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def unitwise_norm(x: torch.Tensor):

    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        '''2 dimensional 인 경우 column방향으로 norm을 구한다.'''
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(object):
    def __init__(self, model_params,
                 clip_threshold: float = 1e-2,
                 eps: float = 1e-3):

        if clip_threshold < 0:
            raise ValueError("The clip_threshold should be larger than zero. Your value : {}".
                             format(clip_threshold))
        if eps < 0:
            raise ValueError("The eps should be larger than 0. Your eps value: {}".format(eps))

        self.params = model_params
        self.eps = eps
        self.clip_threshold = clip_threshold

    @torch.no_grad()
    def Adaptive_clip(self):

        for param in self.params:

            if param.grad is None:
                continue

            '''eps 는 param_norm 값이 eps 보다 작은 경우 해당 위치에 eps 값을 할당한다. 그렇지 않으면 보존~!'''
            param_norm = torch.max(unitwise_norm(param.detach()),
                                   torch.tensor(self.eps).to(param.device)
                                   )

            grad_norm = unitwise_norm(param.grad.detach())  # unitwise_norm -> Frobenious norm for input channel
            clip_numerator = param_norm * self.clip_threshold

            clipping_mask = grad_norm > clip_numerator  # G_F > lambda * W_F

            clipped_grad = param.grad * clip_numerator / (torch.max(grad_norm, torch.tensor(1e-6).to(grad_norm.device)))

            param.grad.detach().data.copy_(torch.where(clipping_mask, clipped_grad, param.grad))


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size_x, kernel_size_y,
                 pad_x, pad_y, dilation_x, dilation_y,
                 stride_x, stride_y, num_groups, bias=True, eps=1e-5):
        super(WSConv2D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(kernel_size_x, kernel_size_y), stride=(stride_x, stride_y),
                                       padding=(pad_x, pad_y), dilation=(dilation_x, dilation_y), groups=num_groups,
                                       bias=bias)
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(self.weight, dim=(1, 2, 3), keepdim=True)
        std = torch.std(self.weight, dim=(1, 2, 3), keepdim=True, unbiased=False)
        standardized_weight = (self.weight - mean) / (std + self.eps)

        return F.conv2d(x, weight=standardized_weight, bias=self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)


class Generator(nn.Module):

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        eps=1e-05
        momentum=1e-3
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8 ** 2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128, eps=eps, momentum=momentum),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=eps, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=eps, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3)
        )

    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)


# if __name__ == '__main__':
#     from torchsummary import summary

#     z = torch.randn((50,100), dtype=torch.float)
#     model = Generator(z_dim=100)

#     model.print_shape(z)
#     images = model(z)
#     print(images.shape)

#     summary(model, input_size=((1, 100)))
