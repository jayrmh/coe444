
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
valid_location = "C:\\Users\Jayroop\Desktop\Security\Subset\\valid"


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


datagen = ImageDataGenerator(rescale=1./255)

valid_batches = datagen.flow_from_directory(valid_location,target_size=(224, 224), classes=['og','steg'], batch_size = 5, class_mode='categorical')

vgg16_model = keras.applications.vgg16.VGG16()



new_model = keras.models.Sequential()
for layer in vgg16_model.layers[:-1]:

    new_model.add(layer)

for layer in  new_model.layers:
    layer.trainable = False

new_model.add(keras.layers.Dense(2, activation = 'softmax'))
new_model.summary()
new_model.load_weights('Detector.h5')
plot_model(new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


val_y = []
count = 0
for i in range(1, 69):
    print(count)
    test1, test2 = next(valid_batches)
    test2 = test2[:,0]
    for j in range(0, 5):
        print(count)
        print(test2[j])
        val_y.append(test2[j])
        count = count + 1

print(val_y)
predictions = new_model.predict_generator(valid_batches, steps=68)
print(np.round(predictions[:,0]))
predictions = np.round(predictions[:,0])

cm_plot_labels = ['og', 'steg']
plot_confusion_matrix(val_y, predictions, cm_plot_labels, title='Confusion Matrix', normalize=True)

plt.show()
