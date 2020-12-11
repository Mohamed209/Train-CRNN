import sys
import os
import cv2
import imgaug.augmenters as iaa
DATA_PATH = sys.argv[1]
AUG_SAMPLES = 4
# Augmentation pipeline
seq = iaa.OneOf([
    # Add gaussian noise
    iaa.AdditiveGaussianNoise(scale=(30, 60)),
    # Motion Blur
    iaa.MotionBlur(angle=(0, 288)),
    # Add a value to all pixels in an image
    iaa.Add((-45, 45)),
    # Making the image darker or brighter
    iaa.Multiply((0.5, 1.5)),
    # Add salt an pepper noise
    iaa.SaltAndPepper((0.03, 0.05)),
    # jpeg compression
    iaa.JpegCompression(compression=(75, 99))
])
data = sorted(os.listdir(DATA_PATH))
for sample in data:
    print("loaded >>>", sample)
    if sample.split('.')[-1] in ['png','jpg','jpeg','tif']:
        img = cv2.imread(DATA_PATH+sample)
        for i in range(AUG_SAMPLES):
            aug = seq.augment_image(image=img)
            cv2.imwrite(DATA_PATH+'aug'+str(i)+sample, aug)
    else:
        with open(DATA_PATH+sample, 'r') as s:
            sent = s.readline()
            for i in range(AUG_SAMPLES):
                with open(DATA_PATH+'aug'+str(i)+sample, mode='w') as l:
                    l.write(sent)
