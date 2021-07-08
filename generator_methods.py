import imageio
import imgaug as ia
from imgaug import augmenters as iaa

STANDARD_SIZE = 74

seq_0 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE)
])

seq_1 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    iaa.GaussianBlur(sigma=0.2),
    iaa.LogContrast(gain=0.75)
])

seq_2 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    # iaa.MedianBlur(k=5),
    iaa.Add(value=-45),
    # iaa.Fog(),
    # iaa.Affine(cval=3)
])

seq_3 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    # iaa.MotionBlur(angle=30),
    iaa.Add(value=-45),
    iaa.GammaContrast(gamma=1.44),
    iaa.pillike.EnhanceSharpness(factor=0.15),
    # iaa.Clouds(),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
])

seq_4 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    iaa.PiecewiseAffine(scale=0.03),
    # iaa.Affine(cval=20),
    iaa.Pepper(p=(0.05, 0.1))
])

seq_5 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    # iaa.AveragePooling(kernel_size=4),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
])

seq_6 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    iaa.GaussianBlur(sigma=0.2),
    iaa.Dropout(p=0.1)
])

seq_7 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    # iaa.imgcorruptlike.Snow(severity=1),
    # iaa.Fog(),
    # iaa.GammaContrast(gamma=1.12)
    # iaa.GammaContrast((0.5, 2.0), per_channel=True)
    iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
    iaa.MedianBlur(k=3),
    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
])

seq_8 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    # iaa.Cartoon(blur_ksize=2, segmentation_size=1.0, saturation=2.0)
    iaa.Add(value=(15, 35), per_channel=0.5),
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
])

seq_9 = iaa.Sequential([
    iaa.Resize(size=STANDARD_SIZE),
    iaa.ImpulseNoise(0.1),
    iaa.Flipud(0.5)
])


def generator(path):
    image = imageio.imread(path)
    image_aug = [
        seq_0(image=image),
        seq_1(image=image),
        seq_2(image=image),
        seq_3(image=image),
        seq_4(image=image),
        seq_5(image=image),
        seq_6(image=image),
        seq_7(image=image),
        seq_8(image=image),
    ]
    return image_aug

    # ia.imshow(seq_7(image=image))

