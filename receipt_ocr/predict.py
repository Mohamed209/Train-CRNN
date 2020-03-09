from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model
import numpy as np
import cv2
import pyarabic.araby as araby
import string
letters = araby.LETTERS+string.printable+'٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩'
test_model = load_model('test_model.h5', compile=False)
test_model.load_weights('ckpts/CRNN--30--2.860.hdf5')

test_image = cv2.imread(
    'test_images/IMG_20190123_155750.jpg', 0)
test_image = cv2.resize(test_image, (432, 32))
test_image = np.expand_dims(test_image, -1)
test_image = np.expand_dims(test_image, axis=0)
prediction = test_model.predict(test_image)
# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                               greedy=True)[0][0])
# see the results
i = 0
for x in out:
    print("predicted text = ", end='')
    for p in x:
        if int(p) != -1:
            print(letters[int(p)], end='')
    print('\n')
    i += 1