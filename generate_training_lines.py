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
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import webtext
import h5py

ara_letters = araby.LETTERS+u' ٠١٢٣٤٥٦٧٨٩,()/\-'
eng_letters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,()/\- '
letters = ara_letters

SHADOW_DISTRIBUTION = [1, 0]
SHADOW_WEIGHT = [0.3, 0.7]
INV_DISTRIBUTION = [1, 0]
INV_WEIGHT = [0.3, 0.7]
FIT = False
SAVE_PATH = 'dataset/generated_data/'

text_size = [50, 60, 70]
blur = [0, 1]
skewing_angle = [0, 1, 2]
background_type = [1, 0]
distorsion_type = [2, 0, 3]
text_color = ["#000000", "#282828", "#505050"]


arabic_words = []
arabic_words_not_reshaped = []
english_words = []
arabic_nums = []
english_nums = []


def generate_words():
    # generate arabic words
    # print("Reading Arabic Corpus :)")
    # with open('dataset/text_corpus/ara_corpus.txt', encoding='utf-8') as f:
    #     for line in tqdm(f.readlines()):
    #         line = line.replace('\n','').strip()
    #         arabic_words_not_reshaped.append(line)
    #         line = arabic_reshaper.reshape(line)
    #         line = get_display(line)
    #         arabic_words.append(line)

    # generate arabic numbers
    print("Reading Arabic Numbers Corpus :)")
    with open('dataset/text_corpus/aranums.txt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.replace('\n', '').strip()
            if len(line) < 1:
                continue
            arabic_nums.append(line)

    # print("Reading English Numbers Corpus :)")
    # with open('dataset/text_corpus/engnums.txt', encoding='utf-8') as f:
    #     for line in tqdm(f.readlines()):
    #         line = line.replace('\n','').strip()
    #         if len(line)<1:
    #             continue
    #         english_nums.append(line)

    # print("Reading English corpus from NLTK :)")
    # english_words.extend([w for w in webtext.words('firefox.txt')])


generate_words()


#gen_count = 100000
nums_count = 3000

# english_generator = GeneratorFromStrings(
#     strings=english_words,
#     language='en',
#     count=gen_count,
#     size=text_size,
#     blur=blur,
#     background_type=background_type,
#     text_color=text_color,
#     distorsion_type = distorsion_type,
#     skewing_angle = skewing_angle
# )
# english_generator_nums = GeneratorFromStrings(
#     strings=english_nums,
#     language='en',
#     count=nums_count,
#     size=text_size,
#     blur=blur,
#     background_type=background_type,
#     text_color=text_color,
#     distorsion_type = distorsion_type,
#     skewing_angle = skewing_angle
# )
# arabic_generator = GeneratorFromStrings(
#     strings=arabic_words,
#     language='ar',
#     count=gen_count,
#     size=text_size,
#     blur=blur,
#     background_type=background_type,
#     text_color=text_color,
#     distorsion_type = distorsion_type,
#     skewing_angle = skewing_angle
# )
arabic_generator_nums = GeneratorFromStrings(
    strings=arabic_nums,
    language='ar',
    count=nums_count,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color,
    distorsion_type=distorsion_type,
    skewing_angle=skewing_angle
)


img_h = 32
img_w = 432
max_label_len = 15


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


if __name__ == "__main__":

    gens = [arabic_generator_nums]
    for i in range(len(gens)):
        for (img, lbl) in tqdm(gens[i]):
            try:
                text_to_labels(lbl)
                adjlabel = get_display(lbl)
                ID = str(uuid.uuid4())
                with open(SAVE_PATH+ID+'.gt', 'w',encoding='utf-8') as l:
                    l.write(adjlabel)
            except ValueError:
                print("contain toxic chars ", [char for char in lbl])
                continue
            npimage = np.array(img)
            if np.random.choice(SHADOW_DISTRIBUTION, p=SHADOW_WEIGHT):
                npimage = add_parallel_light(npimage)
            elif np.random.choice(INV_DISTRIBUTION, p=INV_WEIGHT):
                npimage = 255-npimage
            npimage = cv2.resize(npimage, (img_w, img_h))
            cv2.imwrite(SAVE_PATH+ID+'.png', npimage)
