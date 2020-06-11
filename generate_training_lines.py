import arabic_reshaper
import pyarabic.araby as araby
import cv2
import PIL
import uuid
import re
import random
import numpy as np
from bidi.algorithm import get_display
from trdg.generators import GeneratorFromStrings
from trdg.utils import add_parallel_light
from scipy.stats import norm
from PIL import Image
from tqdm import tqdm
import pyarabic.araby as araby
import string
from multiprocessing import Pool
letters = u'٠١٢٣٤٥٦٧٨٩'+'0123456789'
SHADOW_DISTRIBUTION = [1, 0]
SHADOW_WEIGHT = [0.4, 0.6]
INV_DISTRIBUTION = [1, 0]
INV_WEIGHT = [0.3, 0.7]
FIT = False
SAVE_PATH = 'dataset/generated_data/'
eng_lines = []
ara_lines = []
text_size = [40, 50, 60]
blur = [0, 1]
skewing_angle = [0, 1, 2]
background_type = [1, 0, 2]
distorsion_type = [2, 0, 3]
text_color = ["#000000", "#282828", "#505050"]


def add_fake_shdows(pil_img):
    open_cv_image = np.array(pil_img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    open_cv_image = add_parallel_light(open_cv_image)
    return Image.fromarray(open_cv_image)


def invert(pil_img):
    open_cv_image = np.array(pil_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    inv = cv2.bitwise_not(open_cv_image)
    return Image.fromarray(inv)


def generate_english_lines():
    flag = False
    with open('dataset/text_corpus/engmeters.txt', mode='r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            eng_lines.append(line)


def generate_arabic_lines():
    flag = False
    with open('dataset/text_corpus/arameters.txt', mode='r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            ara_lines.append(line)
        # mixed_lines_no_res.extend(mixed_lines)
        # for i in range(len(mixed_lines)):
        #     mixed_lines[i] = arabic_reshaper.reshape(mixed_lines[i])
        #     mixed_lines[i] = get_display(mixed_lines[i])


generate_arabic_lines()
generate_english_lines()

english_generator = GeneratorFromStrings(
    strings=eng_lines,
    language='en',
    count=25,
    size=text_size,
    distorsion_type=distorsion_type,
    skewing_angle=skewing_angle,
    blur=blur,
    background_type=background_type,
    text_color=text_color
)
arabic_generator = GeneratorFromStrings(
    strings=ara_lines,
    language='ar',
    count=25,
    size=text_size,
    distorsion_type=distorsion_type,
    skewing_angle=skewing_angle,
    blur=blur,
    background_type=background_type,
    text_color=text_color
)


def save_eng_lines(img, lbl):
    if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
        img = add_fake_shdows(img)
    elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
        img = invert(img)
    img = img.resize((432, 32), Image.ANTIALIAS)
    ID = str(uuid.uuid4())
    img.save(SAVE_PATH+ID+'.png')
    with open(SAVE_PATH+ID+'.txt', 'w', encoding='utf-8') as label:
        label.writelines(lbl)


def save_ara_lines(img, lbl):
    if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
        img = add_fake_shdows(img)
    elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
        img = invert(img)
    img = img.resize((432, 32), Image.ANTIALIAS)
    ID = str(uuid.uuid4())
    img.save(SAVE_PATH+ID+'.png')
    with open(SAVE_PATH+ID+'.txt', 'w', encoding='utf-8') as label:
        label.writelines(lbl)


if __name__ == "__main__":
    with Pool() as pool:
        pool.starmap(save_ara_lines, [(img, lbl)
                                      for (img, lbl) in tqdm(arabic_generator)])
    with Pool() as pool:
        pool.starmap(save_eng_lines, [(img, lbl)
                                      for (img, lbl) in tqdm(english_generator)])
