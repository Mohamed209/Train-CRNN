from keras.preprocessing.sequence import pad_sequences
import numpy as np
import h5py
import cv2
import os
import string
import pyarabic.araby as araby
import sys
# utils
letters = u'٠١٢٣٤٥٦٧٨٩'+'0123456789'


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
images = np.zeros(shape=(len(data)//2, img_h, img_w, 1))
label_length = np.zeros((len(data)//2, 1), dtype=np.int64)
text = []
i = 0
j = 0
for sample in data:
    print("loaded >>>", sample)
    if sample.split('.')[-1] == 'png':
        img = cv2.imread(DATA_PATH+sample, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_w, img_h))
        img = img.astype(np.float32)
        img = (img / 255.0)
        img = np.expand_dims(img, axis=-1)
        images[i] = img
        i += 1
    else:
        with open(DATA_PATH+sample, 'r',encoding='utf-8') as s:
            sent = s.readlines()
            text.append(sent)
            label_length[j] = len(sent[0])
        j += 1

# we need number repr for text
gt_text = []
textnum = []
for line in text:
    data = line[0].strip()
    textnum.append(text_to_labels(data))
for i in range(len(textnum)):
    gt_text.append(textnum[i])

gt_padded_txt = pad_sequences(
    gt_text, maxlen=8, padding='post', truncating='post', value=0)

print("images >>", images.shape)
print("text >>", gt_padded_txt.shape)
print("label length>>", label_length.shape)

# save np arrays to hard disk so as not to generate them from scratch in the begining of each training session
h5 = h5py.File('../dataset/dataset.h5', 'w')
h5.create_dataset('images', data=images)
h5.create_dataset('text', data=gt_padded_txt)
h5.create_dataset('label_length', data=label_length)
h5.close()
print("np arrays saved to hard disk :)")
