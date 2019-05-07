
import keras
import os
import math
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
import pickle
import gc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import save_model, load_model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from keras import backend as K
import itertools
from keras import optimizers
from sklearn import preprocessing
import keras
import matplotlib.pyplot as plt
from keras.models import save_model, load_model
from sklearn.manifold import TSNE
from sklearn.externals import joblib
from keras.utils.vis_utils import plot_model

test_location = "C:\\Users\Jayroop\Desktop\Security\Subset\\test"



datagen = ImageDataGenerator(rescale=1./255)

test_batches = datagen.flow_from_directory(test_location,target_size=(224, 224), classes=['og','steg'], batch_size = 1, class_mode='categorical')

vgg16_model = keras.applications.vgg16.VGG16()



new_model = keras.models.Sequential()
for layer in vgg16_model.layers[:-1]:

    new_model.add(layer)

for layer in  new_model.layers:
    layer.trainable = False

new_model.add(keras.layers.Dense(2, activation = 'softmax'))
new_model.summary()
new_model.load_weights('Detector.h5')

x,y = next(test_batches)
for i in range(0,5):
    plt.figure(figsize=(2, 2))

    image_label = os.path.dirname(test_batches.filenames[i]) # only OK if shuffle=false
    print(image_label)



predictions = new_model.predict_generator(test_batches, steps=5)
predictions = np.round(predictions[:,0])
print(predictions)
