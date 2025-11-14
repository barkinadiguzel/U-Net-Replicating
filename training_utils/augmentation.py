import torch
import torchvision.transforms.functional as TF
import random

# Elastic deformation is in the original U-Net paper, but not implemented here.
# Only random rotation and shift are applied.
class UNetAugmentation:
    def __init__(self, max_rotation=15, max_shift=0.1):
        self.max_rotation = max_rotation
        self.max_shift = max_shift

    def __call__(self, image, mask):
        # image: CxHxW tensor, mask: 1xHxW tensor

        # random rotation
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # random shift (affine translate)
        max_dx = self.max_shift * image.shape[2]
        max_dy = self.max_shift * image.shape[1]
        translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        image = TF.affine(image, angle=0, translate=translations, scale=1.0, shear=0,
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=0, translate=translations, scale=1.0, shear=0,
                         interpolation=TF.InterpolationMode.NEAREST)

        return image, mask
