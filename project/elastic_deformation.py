import torch
import torchvision.transforms.v2 as transforms

def get_displacement(alpha, sigma, size):
    dx = torch.rand([1, 1] + size) * 2 - 1
    if sigma > 0.0:
        kx = int(8 * sigma + 1)
        # if kernel size is even we have to make it odd
        if kx % 2 == 0:
            kx += 1
        dx = transforms.functional.gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha / size[0]

    dy = torch.rand([1, 1] + size) * 2 - 1
    if sigma > 0.0:
        ky = int(8 * sigma + 1)
        # if kernel size is even we have to make it odd
        if ky % 2 == 0:
            ky += 1
        dy = transforms.functional.gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha / size[1]
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2