from typing import Callable, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F


def add_label_centered(
    img: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    thickness: int = 2,
    alignment: str = "top",
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, font_scale, thickness=thickness)[0]
    img = img.astype(np.uint8).copy()

    if alignment == "top":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, 50),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    elif alignment == "bottom":
        cv2.putText(
            img,
            text,
            ((img.shape[1] - textsize[0]) // 2, img.shape[0] - textsize[1]),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        raise ValueError("Unknown text alignment")

    return img

def tensor2rgbjet(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    return cv2.applyColorMap(tensor2rgb(tensor, x_max=x_max, x_min=x_min), cv2.COLORMAP_JET)


def tensor2rgb(
    tensor: th.Tensor, x_max: Optional[float] = None, x_min: Optional[float] = None
) -> np.ndarray:
    x = tensor.data.cpu().numpy()
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()

    gain = 255 / np.clip(x_max - x_min, 1e-3, None)
    x = (x - x_min) * gain
    x = x.clip(0.0, 255.0)
    x = x.astype(np.uint8)
    return x


def tensor2image(
    tensor: th.Tensor,
    x_max: Optional[float] = 1.0,
    x_min: Optional[float] = 0.0,
    mode: str = "rgb",
    mask: Optional[th.Tensor] = None,
    label: Optional[str] = None,
) -> np.ndarray:

    tensor = tensor.detach()

    # Apply mask
    if mask is not None:
        tensor = tensor * mask

    if len(tensor.size()) == 2:
        tensor = tensor[None]

    # Make three channel image
    assert len(tensor.size()) == 3, tensor.size()
    n_channels = tensor.shape[0]
    if n_channels == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif n_channels != 3:
        raise ValueError(f"Unsupported number of channels {n_channels}.")

    # Convert to display format
    img = tensor.permute(1, 2, 0)

    if mode == "rgb":
        img = tensor2rgb(img, x_max=x_max, x_min=x_min)
    elif mode == "jet":
        # `cv2.applyColorMap` assumes input format in BGR
        img[:, :, :3] = img[:, :, [2, 1, 0]]
        img = tensor2rgbjet(img, x_max=x_max, x_min=x_min)
        # convert back to rgb
        img[:, :, :3] = img[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"Unsupported mode {mode}.")

    if label is not None:
        img = add_label_centered(img, label)

    return img
    
# d: b x 1 x H x W
# screenCoords: b x 2 x H X W
# focal: b x 2 x 2
# princpt: b x 2
# out: b x 3 x H X W
def depthImgToPosCam_Batched(d, screenCoords, focal, princpt):
    p = screenCoords - princpt[:, :, None, None]
    x = (d * p[:, 0:1, :, :]) / focal[:, 0:1, 0, None, None]
    y = (d * p[:, 1:2, :, :]) / focal[:, 1:2, 1, None, None]
    return th.cat([x, y, d], dim=1)

# p: b x 3 x H x W
# out: b x 3 x H x W
def computeNormalsFromPosCam_Batched(p):
    p = F.pad(p, (1, 1, 1, 1), "replicate")
    d0 = p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]
    d1 = p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]
    n = th.cross(d0, d1, dim=1)
    norm = th.norm(n, dim=1, keepdim=True)
    norm = norm + 1e-5
    norm[norm < 1e-5] = 1  # Can not backprop through this
    return -n / norm

def visualize_normal(inputs, depth_p):
    # Normals
    uv = th.stack(
        th.meshgrid(
            th.arange(depth_p.shape[2]), th.arange(depth_p.shape[1]), indexing="xy"
        ),
        dim=0,
    )[None].float().cuda()
    position = depthImgToPosCam_Batched(
        depth_p[None, ...], uv, inputs["focal"], inputs["princpt"]
    )
    normal = 0.5 * (computeNormalsFromPosCam_Batched(position) + 1.0)
    normal = normal[0, [2, 1, 0], :, :]  # legacy code assumes BGR format
    normal_p = tensor2image(normal, label="normal_p")

    return normal_p