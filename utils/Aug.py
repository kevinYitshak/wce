from albumentations import (CLAHE, Compose, ElasticTransform, GridDistortion,
                            HorizontalFlip, OneOf, OpticalDistortion,
                            RandomBrightnessContrast, RandomGamma,
                            RandomRotate90, ShiftScaleRotate, Transpose,
                            VerticalFlip)
                            
class Augmentation(object):

    def __init__(self):
        super(Augmentation, self).__init__()

        self._hflip = HorizontalFlip(p=0.5)
        self._vflip = VerticalFlip(p=0.5)
        self._clahe = CLAHE(p=.3)
        self._rotate = RandomRotate90(p=.3)
        self._brightness = RandomBrightnessContrast(p=0.3)
        self._gamma = RandomGamma(p=0.3)
        self._transpose = Transpose(p=0.3)
        self._elastic = ElasticTransform(
            p=.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        self._distort = GridDistortion(p=0.3)
        self._affine = ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3)

    def _aug(self):
        aug = [self._hflip, self._vflip, self._clahe, self._rotate, self._brightness,
                self._gamma, self._transpose, self._elastic, self._distort, self._affine]
        
        return Compose(aug)