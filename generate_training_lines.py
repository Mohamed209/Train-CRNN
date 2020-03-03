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
from data_loader import pull_wikipedia_content
from tqdm import tqdm
from joblib import Parallel, delayed
import pyarabic.araby as araby
import string
from multiprocessing import Pool
letters = araby.LETTERS+string.printable+u'٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩'
SHADOW_DISTRIBUTION = [1, 0]
SHADOW_WEIGHT = [0.3, 0.7]
INV_DISTRIBUTION = [1, 0]
INV_WEIGHT = [0.3, 0.7]
FIT = False
SAVE_PATH = 'dataset/generated_data/'
ara_lines = []
eng_lines = []
ara_lines_no_res = []
mixed_lines = []
mixed_lines_no_res = []
text_size = [70, 80, 90]
blur = [0, 1]
skewing_angle = [0, 1, 2]
background_type = [0, 1, 2]
distorsion_type = [0, 2, 3]
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
    with open('dataset/text_corpus/eng_gt.txt') as f:
        for line in tqdm(f.readlines()):
            if line.strip():
                for ch in list(set(line)):
                    if ch not in letters:
                        flag = True
                        print("unwanted char ", ch)
                        break
                if flag:
                    flag = False
                    continue
                eng_lines.append(line)
        random.shuffle(eng_lines)


def generate_mixed_lines():
    flag = False
    with open('dataset/text_corpus/db_gt.txt', mode='r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            if line.strip():
                for ch in list(set(line)):
                    if ch not in letters:
                        flag = True
                        print("unwanted char ", ch)
                        break
                if flag:
                    flag = False
                    continue
                mixed_lines.append(line)
        random.shuffle(mixed_lines)
        mixed_lines_no_res.extend(mixed_lines)
        for i in range(len(mixed_lines)):
            mixed_lines[i] = arabic_reshaper.reshape(mixed_lines[i])
            mixed_lines[i] = get_display(mixed_lines[i])


####################################################################
'''
create N generators and randomly select one for each iteration
'''
####################################################################
generate_mixed_lines()
generate_english_lines()
english_generator = GeneratorFromStrings(
    strings=eng_lines,
    language='en',
    count=250000,
    size=np.random.choice(text_size),
    distorsion_type=np.random.choice(distorsion_type),
    skewing_angle=np.random.choice(skewing_angle),
    blur=np.random.choice(blur),
    background_type=np.random.choice(background_type),
    text_color=np.random.choice(text_color)
)
mixed_generator = GeneratorFromStrings(
    strings=mixed_lines,
    language='mix',
    count=250000,
    size=np.random.choice(text_size),
    distorsion_type=np.random.choice(distorsion_type),
    skewing_angle=np.random.choice(skewing_angle),
    blur=np.random.choice(blur),
    background_type=np.random.choice(background_type),
    text_color=np.random.choice(text_color)
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


def save_mixed_lines(img, lbl):
    if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
        img = add_fake_shdows(img)
    elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
        img = invert(img)
    img = img.resize((432, 32), Image.ANTIALIAS)
    ID = str(uuid.uuid4())
    img.save(SAVE_PATH+ID+'.png')
    with open(SAVE_PATH+ID+'.txt', 'w', encoding='utf-8') as label:
        label.writelines(mixed_lines_no_res[mixed_lines.index(lbl)])


if __name__ == "__main__":
    with Pool() as pool:
        pool.starmap(save_mixed_lines, [(img, lbl)
                                        for (img, lbl) in tqdm(mixed_generator)])
    with Pool() as pool:
        pool.starmap(save_mixed_lines, [(img, lbl)
                                        for (img, lbl) in tqdm(mixed_generator)])
