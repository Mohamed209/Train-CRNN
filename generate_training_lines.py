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
import multiprocessing as mp
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
text_size = [80, 90, 100]
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
    # pull_wikipedia_content(language='english')
    with open('dataset/text_corpus/eng_gt.txt') as f:
        for line in f.readlines():
            if line.strip():
                eng_lines.append(line)
        random.shuffle(eng_lines)


def generate_mixed_lines():
    with open('dataset/text_corpus/db_gt.txt', mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip():
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

generate_english_lines()
generate_mixed_lines()
english_generator = GeneratorFromStrings(
    strings=eng_lines,
    language='en',
    count=150000,
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
    count=200000,
    size=np.random.choice(text_size),
    distorsion_type=np.random.choice(distorsion_type),
    skewing_angle=np.random.choice(skewing_angle),
    blur=np.random.choice(blur),
    background_type=np.random.choice(background_type),
    text_color=np.random.choice(text_color)
)


def save_lines(img, lbl):
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
    pool = mp.Pool(mp.cpu_count())
    print("started generating english lines :)")
    [pool.apply(save_lines, args=(img, lbl))
     for img, lbl in tqdm(english_generator)]
    print("started generating arabic lines :)")
    [pool.apply(save_lines, args=(img, lbl))
     for img, lbl in tqdm(mixed_generator)]
