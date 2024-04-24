import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image
from PIL import Image
from typing import Union


class DepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
    
    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)
    
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
    
    def _infer(self, x: torch.Tensor):
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        return self(x)['metric_depth']
    
    def _infer_with_pad_aug(self, x: torch.Tensor, pad_input: bool=True, fh: float=3, fw: float=3, upsampling_mode: str='bicubic', padding_mode="reflect", **kwargs) -> torch.Tensor:

        # assert x is nchw and c = 3
        assert x.dim() == 4, "x must be 4 dimensional, got {}".format(x.dim())
        assert x.shape[1] == 3, "x must have 3 channels, got {}".format(x.shape[1])

        if pad_input:
            assert fh > 0 or fw > 0, "atlease one of fh and fw must be greater than 0"
            pad_h = int(np.sqrt(x.shape[2]/2) * fh)
            pad_w = int(np.sqrt(x.shape[3]/2) * fw)
            padding = [pad_w, pad_w]
            if pad_h > 0:
                padding += [pad_h, pad_h]
            
            x = F.pad(x, padding, mode=padding_mode, **kwargs)
        out = self._infer(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode=upsampling_mode, align_corners=False)
        if pad_input:
            # crop to the original size, handling the case where pad_h and pad_w is 0
            if pad_h > 0:
                out = out[:, :, pad_h:-pad_h,:]
            if pad_w > 0:
                out = out[:, :, :, pad_w:-pad_w]
        return out
    
    def infer_with_flip_aug(self, x, pad_input: bool=True, **kwargs) -> torch.Tensor:

        # infer with horizontal flip and average
        out = self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)
        out_flip = self._infer_with_pad_aug(torch.flip(x, dims=[3]), pad_input=pad_input, **kwargs)
        out = (out + torch.flip(out_flip, dims=[3])) / 2
        return out
    
    def infer(self, x, pad_input: bool=True, with_flip_aug: bool=True, **kwargs) -> torch.Tensor:
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
            with_flip_aug (bool, optional): whether to use horizontal flip augmentation. Defaults to True.
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        if with_flip_aug:
            return self.infer_with_flip_aug(x, pad_input=pad_input, **kwargs)
        else:
            return self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)
    
    @torch.no_grad()
    def infer_pil(self, pil_img, pad_input: bool=True, with_flip_aug: bool=True, output_type: str="numpy", **kwargs) -> Union[np.ndarray, PIL.Image.Image, torch.Tensor]:
        """
        Inference interface for the model for PIL image
        Args:
            pil_img (PIL.Image.Image): input PIL image
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
            with_flip_aug (bool, optional): whether to use horizontal flip augmentation. Defaults to True.
            output_type (str, optional): output type. Supported values are 'numpy', 'pil' and 'tensor'. Defaults to "numpy".
        """
        x = transforms.ToTensor()(pil_img).unsqueeze(0).to(self.device)
        out_tensor = self.infer(x, pad_input=pad_input, with_flip_aug=with_flip_aug, **kwargs)
        if output_type == "numpy":
            return out_tensor.squeeze().cpu().numpy()
        elif output_type == "pil":
            # uint16 is required for depth pil image
            out_16bit_numpy = (out_tensor.squeeze().cpu().numpy()*256).astype(np.uint16)
            return Image.fromarray(out_16bit_numpy)
        elif output_type == "tensor":
            return out_tensor.squeeze().cpu()
        else:
            raise ValueError(f"output_type {output_type} not supported. Supported values are 'numpy', 'pil' and 'tensor'")
    