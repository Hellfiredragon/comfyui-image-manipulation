from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
import torch


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

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

