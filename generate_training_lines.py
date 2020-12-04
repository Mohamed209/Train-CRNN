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

ara_letters = araby.LETTERS+u' ٠١٢٣٤٥٦٧٨٩'
eng_letters = string.printable
letters = ara_letters+eng_letters

SHADOW_DISTRIBUTION = [1, 0]
SHADOW_WEIGHT = [0.4, 0.6]
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


arabic_words =[]
arabic_words_not_reshaped =[]
english_words = []
arabic_nums = []
english_nums = []

def generate_words():
    # generate arabic words
    print("Reading Arabic Corpus :)")
    with open('dataset/text_corpus/ara_corpus.txt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()[:2000]):
            line = line.replace('\n','').strip()
            arabic_words_not_reshaped.append(line)
            line = arabic_reshaper.reshape(line)
            line = get_display(line)
            arabic_words.append(line)
    
    # generate arabic numbers
    print("Reading Arabic Numbers Corpus :)")
    with open('dataset/text_corpus/aranums.txt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()[:2000]):
            line = line.replace('\n','').strip()
            if len(line)<1:
                continue
            arabic_nums.append(line)

    print("Reading English Numbers Corpus :)")
    with open('dataset/text_corpus/engnums.txt', encoding='utf-8') as f:
        for line in tqdm(f.readlines()[:2000]):
            line = line.replace('\n','').strip()
            if len(line)<1:
                continue
            english_nums.append(line)

    print("Reading English corpus from NLTK :)")
    english_words.extend([w for w in webtext.words('firefox.txt')])
    


generate_words()


gen_count = 100000
nums_count = 50000
english_generator = GeneratorFromStrings(
    strings=english_words,
    language='en',
    count=gen_count,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color,
    distorsion_type = distorsion_type,
    skewing_angle = skewing_angle
)
english_generator_nums = GeneratorFromStrings(
    strings=english_nums,
    language='en',
    count=nums_count,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color,
    distorsion_type = distorsion_type,
    skewing_angle = skewing_angle
)
arabic_generator = GeneratorFromStrings(
    strings=arabic_words,
    language='ar',
    count=gen_count,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color,
    distorsion_type = distorsion_type,
    skewing_angle = skewing_angle
)
arabic_generator_nums = GeneratorFromStrings(
    strings=arabic_nums,
    language='ar',
    count=nums_count,
    size=text_size,
    blur=blur,
    background_type=background_type,
    text_color=text_color,
    distorsion_type = distorsion_type,
    skewing_angle = skewing_angle
)


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

img_h = 32
img_w = 128
max_label_len = 15
images = []
label_length = []
labels = []

if __name__ == "__main__":

    gens = [arabic_generator,english_generator,arabic_generator_nums,english_generator_nums]
    for i in range(len(gens)):
        for (img,lbl) in tqdm(gens[i]):
            if i ==0:
                try:
                    labels.append(text_to_labels(arabic_words_not_reshaped[arabic_words.index(lbl)]))
                except ValueError:
                    print("contain toxic chars ",[char for char in lbl])
                    continue
            else:
                labels.append(text_to_labels(lbl))
            npimage = np.array(img)
            npimage = cv2.cvtColor(npimage, cv2.COLOR_BGR2GRAY)
            npimage = cv2.resize(npimage,(img_w,img_h))
            npimage = npimage.astype(np.float32)
            npimage = (npimage / 255.0)
            npimage = np.expand_dims(npimage, axis=-1)
            images.append(npimage)
            label_length.append(len(lbl))
    
    gt_padded_txt = pad_sequences(labels, maxlen=max_label_len, padding='post', truncating='post', value=letters.index('\t'))
    
    images_tensor = np.array(images)
    
    label_length_tensor = np.array(label_length).astype(np.int64)

    print("finished generating .. check tensor shapes")
    print("images {} label_lenght {} label {}".format(images_tensor.shape,label_length_tensor.shape,gt_padded_txt.shape))


    # for i in range(len(images_tensor)):
    #     print(labels_to_text(gt_padded_txt[i]))
    #     disp = images_tensor[i]*255.0
    #     cv2.imwrite('dataset/generated_data/'+str(i)+'.png',disp)

    # save np arrays to hard disk so as not to generate them from scratch in the begining of each training session
    h5 = h5py.File('dataset/dataset.h5', 'w')
    h5.create_dataset('images', data=images_tensor)
    h5.create_dataset('text', data=gt_padded_txt)
    h5.create_dataset('label_length', data=label_length_tensor)
    h5.close()
    print("np arrays saved to hard disk :)")
