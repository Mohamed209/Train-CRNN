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

ara_letters = araby.LETTERS+u' ٠١٢٣٤٥٦٧٨٩'
eng_letters = string.printable

SHADOW_DISTRIBUTION = [1, 0]
SHADOW_WEIGHT = [0.4, 0.6]
INV_DISTRIBUTION = [1, 0]
INV_WEIGHT = [0.3, 0.7]
FIT = False
SAVE_PATH = 'dataset/generated_data/'

arawords = []
engwords = []

text_size = [40, 50, 60]
blur = [0, 1]
#skewing_angle = [0, 1, 2]
background_type = [1, 0]
#distorsion_type = [2, 0, 3]
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


def generate_words():
    with open('dataset/text_corpus/ftc.txt', mode='r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            if len(line) <3:
                continue
            try :
                line.encode('ascii')
                engwords.append(line)
            except UnicodeEncodeError :
                line = arabic_reshaper.reshape(line)
                line = get_display(line)
                arawords.append(line)


generate_words()

english_generator = GeneratorFromStrings(
    strings=engwords,
    language='en',
    count=25000,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color
)
arabic_generator = GeneratorFromStrings(
    strings=arawords,
    language='ar',
    count=25000,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color
)


def save_eng_lines(img, lbl):
    # if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
    #     img = add_fake_shdows(img)
    # elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
    #     img = invert(img)
    img = img.resize((128, 32), Image.ANTIALIAS)
    ID = str(uuid.uuid4())
    img.save(SAVE_PATH+ID+'.png')
    with open(SAVE_PATH+ID+'.txt', 'w', encoding='utf-8') as label:
        label.write(lbl)


def save_ara_lines(img, lbl):
    # if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
    #     img = add_fake_shdows(img)
    # elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
    #     img = invert(img)
    img = img.resize((128, 32), Image.ANTIALIAS)
    ID = str(uuid.uuid4())
    img.save(SAVE_PATH+ID+'.png')
    if any(ch in u'٠١٢٣٤٥٦٧٩' for ch in lbl):
        lbl=lbl
    else:
        lbl=lbl[::-1]
    with open(SAVE_PATH+ID+'.txt', 'w', encoding='utf-8') as label:
        label.write(lbl)


if __name__ == "__main__":
    with Pool() as pool:
        pool.starmap(save_ara_lines, [(img, lbl)
                                      for (img, lbl) in tqdm(arabic_generator)])
    with Pool() as pool:
        pool.starmap(save_eng_lines, [(img, lbl)
                                      for (img, lbl) in tqdm(english_generator)])
