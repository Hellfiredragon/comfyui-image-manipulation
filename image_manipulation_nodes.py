from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
import torch
from torch import Tensor
import kornia.filters

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# PIL to Mask
def pil2mask(image):
    return pil2tensor(image)


# Mask to PIL
def mask2pil(mask):
    return tensor2pil(mask).convert('L')


def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def binary_dilation(mask: Tensor, radius: int):
    kernel = torch.ones(1, radius * 2 + 1, device=mask.device)
    mask = kornia.filters.filter2d_separable(mask, kernel, kernel, border_type="constant")
    mask = (mask > 0).to(mask.dtype)
    return mask


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma))


def make_odd(x):
    if x > 0 and x % 2 == 0:
        return x + 1
    return x


class AlphaApplyMaskToImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "image_blend_mask"
    CATEGORY = "IM Pack"

    def image_blend_mask(self, image, mask):

        # # Convert images to PIL
        image = tensor2pil(image)
        mask = mask2pil(mask)

        image.putalpha(mask)

        return (pil2tensor(image), pil2mask(mask))


def hex_to_rgb(hex_color: str):
    """
    Convert a hex color (e.g., "#00ff00") to an (R, G, B) tuple.
    """
    hex_color = hex_color.strip()
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


BODY_PARTS = {
    "face-left": "faeb19",
    "face-right": "f9fb0e",
    "torso-front": "1450c2",
    "torso-back": "253ca3",
    "upper-left-arm-outside": "d8ba56",
    "upper-left-arm-inside": "aabd69",
    "upper-right-arm-outside": "c0bc60",
    "upper-right-arm-inside": "91bf74",
    "lower-left-arm-outside": "fbdc24",
    "lower-left-arm-inside": "f0c73c",
    "lower-right-arm-outside": "e4c04a",
    "lower-right-arm-inside": "fcce2e",
    "right-hand": "086edd",
    "left-hand": "0461df",
    "upper-left-leg-front": "06a6c6",
    "upper-left-leg-back": "0f90d0",
    "upper-right-leg-front": "16adb9",
    "upper-right-leg-back": "0b9ccb",
    "lower-left-leg-front": "56bb90",
    "lower-left-leg-back": "26b3ac",
    "lower-right-leg-front": "72bd82",
    "lower-right-leg-back": "37b99f",
    "right-foot": "0e7ad8",
    "left-foot": "1484d4",
}



class CreateMaskFromColorsNode:
    """
    ComfyUI custom node that takes an input image and creates a white mask by selecting
    pixels that match one or more specified colors (given as hex strings) within a tolerance.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        

        return {
            "required": {
                "image": ('IMAGE', {}),
                # Provide a comma-separated list of hex colors, e.g., "#00ff00,#ff0000"
                **{body_part: ("BOOLEAN", {"default": "False"}) for body_part in BODY_PARTS},
                "grow": ("INT", {"default": 16, "min": 0, "max": 8096, "step": 1}),
                "blur": ("INT", {"default": 7, "min": 0, "max": 8096, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_white_mask_from_colors"
    CATEGORY = "IM Pack"

    def create_white_mask_from_colors(self, image, grow, blur, **selected_parts):
        """
        Creates a white mask by selecting pixels in the input image that match any of the specified colors.
        
        Args:
            image (PIL.Image): The input image (assumed to be in RGB mode).
            colors (str): A comma-separated list of hex color strings (e.g., "#00ff00,#ff0000").
            tolerance (int): The allowed per-channel tolerance when matching colors (0 for an exact match).
        
        Returns:
            PIL.Image: A binary (mode "L") mask image where matching pixels are white (255).
        """
        # Parse the comma-separated hex colors and convert them to RGB tuples.
        color_list = [hex_to_rgb(BODY_PARTS[name]) for name, selected in selected_parts.items() if selected]
        
        image = tensor2pil(image)
        
        # Convert the input image to RGB and then to a NumPy array.
        img = np.array(image.convert("RGB"))
        
        # Initialize a boolean mask (same height and width as the input image).
        mask = np.zeros(img.shape[:2], dtype=bool)
        
        # For each target color, update the mask if the pixel is within tolerance.
        for target in color_list:
            target_array = np.array(target, dtype=np.uint8)
            # Compute the absolute difference per channel.
            diff = np.abs(img.astype(np.int16) - target_array.astype(np.int16))
            # A pixel matches if all channel differences are within tolerance.
            current_mask = np.all(diff <= 0, axis=-1)
            mask |= current_mask
        
        # Convert the boolean mask to an 8-bit image: 0 for False, 255 for True.
        white_mask = (mask.astype(np.uint8)) * 255
        
        mask = pil2mask(Image.fromarray(white_mask, mode="L"))

        mask = mask_unsqueeze(mask)
        if grow > 0:
            mask = binary_dilation(mask, grow)
        if blur > 0:
            mask = gaussian_blur(mask, make_odd(blur))
        return (mask.squeeze(1),)

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

