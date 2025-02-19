from .image_manipulation_nodes import AlphaApplyMaskToImage, CreateMaskFromColorsNode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "AlphaApplyMaskToImage": AlphaApplyMaskToImage,
    "CreateMaskFromColorsNode": CreateMaskFromColorsNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AlphaApplyMaskToImage": "Apply Mask To Image (Alpha)",
    "CreateMaskFromColorsNode": "Create Mask From Colors",
}
