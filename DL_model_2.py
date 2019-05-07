import keras
import os
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

import itertools
from keras import optimizers
from sklearn import preprocessing
import keras
import matplotlib.pyplot as plt
from keras.models import save_model, load_model
from sklearn.manifold import TSNE
from sklearn.externals import joblib


train_location = "C:\\Users\Jayroop\Desktop\Security\Subset\\train"
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

train_batches = datagen.flow_from_directory(train_location,target_size=(224, 224), classes=['og','steg'], batch_size = 5, class_mode='categorical')
valid_batches = datagen.flow_from_directory(valid_location,target_size=(224, 224), classes=['og','steg'], batch_size = 5, class_mode='categorical')


vgg16_model = keras.applications.vgg16.VGG16()

model = keras.models.Sequential()
for layer in vgg16_model.layers[:-1]:

    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(keras.layers.Dense(2, activation = 'softmax'))
model.summary()

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

neural_network = model.fit_generator(train_batches, steps_per_epoch=684, validation_data=valid_batches, validation_steps=68, epochs=8)

print(max(neural_network.history['val_acc']))
print(min(neural_network.history['val_loss']))
print(neural_network.history)

plt.plot(neural_network.history['acc'], 'b--', label='Training Accuracy')
plt.plot(neural_network.history['val_acc'], 'r--', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('No of Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(neural_network.history['loss'], 'g--', label='Training Loss')
plt.plot(neural_network.history['val_loss'], 'y--', label='Validation Loss')
plt.title('Training and Validation Accuracy')
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#save the model
model.save_weights('Detector.h5')
model.save('Model.h5')
save_model('Detector.models')
