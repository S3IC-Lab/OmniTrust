import torch
import torch.fft as fft


def Fourier_filter(x, threshold, scale): # 5 , 0.2/0
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.full((B, C, H, W), scale).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered