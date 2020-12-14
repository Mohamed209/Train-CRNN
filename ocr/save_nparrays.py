from keras.preprocessing.sequence import pad_sequences
import numpy as np
import h5py
import cv2
import os
import string
import pyarabic.araby as araby
import sys
from bidi.algorithm import get_display
# utils
letters = u' ٠١٢٣٤٥٦٧٨٩,()/\-'


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))


# data loader
img_h = 32
img_w = 432
# data loader script expects data to be found in folder as pairs of images , txt files contain labels
DATA_PATH = '../dataset/generated_data/'
data = sorted(os.listdir(DATA_PATH))
images = []
label_length = []
text = []

for sample in data:
    if sample.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'tif']:
        with open(DATA_PATH+sample.split('.')[0]+'.gt', 'r', encoding='utf-8') as s:
            word = s.readline()
            try:
                text_to_labels(word)
                #word = get_display(word)
                print(word)
                text.append(word)
                label_length.append(len(word))
                img = cv2.imread(DATA_PATH+sample, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_w, img_h))
                img = img.astype(np.float32)
                img = (img / 255.0)
                img = np.expand_dims(img, axis=-1)
                images.append(img)
            except ValueError:
                continue

# we need number repr for text
gt_text = []
textnum = []
for line in text:
    data = line.strip()
    gt_text.append(text_to_labels(data))

gt_padded_txt = pad_sequences(
    gt_text, maxlen=15, padding='post', truncating='post', value=0)

images_tensor = np.array(images, dtype=np.float32)
label_length_tensor = np.array(label_length, dtype=np.int64)

print("images >>", images_tensor.shape)
print("text >>", gt_padded_txt.shape)
print("label length>>", label_length_tensor.shape)

# save np arrays to hard disk so as not to generate them from scratch in the begining of each training session
h5 = h5py.File('../dataset/dataset.h5', 'w')
h5.create_dataset('images', data=images)
h5.create_dataset('text', data=gt_padded_txt)
h5.create_dataset('label_length', data=label_length)
h5.close()
print("np arrays saved to hard disk :)")
