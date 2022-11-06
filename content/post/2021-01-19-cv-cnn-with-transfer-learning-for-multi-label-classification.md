---
title: CV - CNN with Transfer Learning for Multi-Class Classification
author: Michael Fuchs
date: '2021-01-19'
slug: cv-cnn-with-transfer-learning-for-multi-label-classification
categories:
  - R
tags:
  - R Markdown
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---


# 1 Introduction

After showing how to build [binary classification models using transfer learning](https://michael-fuchs-python.netlify.app/2021/01/17/computer-vision-cnn-with-transfer-learning/), I would like to show this again for multiple classification problems. 

Once again, I used the [Keras Application](https://keras.io/api/applications/) [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) for transfer learning. 

For this publication I used the images from the *Animal Faces* dataset from the statistics platform ["Kaggle"](https://www.kaggle.com/andrewmvd/animal-faces). You can download the used data from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/Computer%20Vision/CNN%20for%20Multi%20Class%20Classification). 


# 2 Import the libraries



```r
from preprocessing_multi_CNN import Train_Validation_Test_Split

import numpy as np
import pandas as pd

import os
import shutil

import pickle as pk

import cv2 
import matplotlib.pyplot as plt
%matplotlib inline 

from keras import layers
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from keras.applications import VGG19
```


# 3 Data pre-processing

How the data for a CNN model training must be prepared I have already explained in this post: [Computer Vision - CNN for Multi-Class Classification](https://michael-fuchs-python.netlify.app/2021/01/15/computer-vision-cnn-for-multi-label-classification/)

The exact functionality of the code I have explained in this post: [Automate the Boring Stuff](https://michael-fuchs-python.netlify.app/2021/01/01/computer-vision-automate-the-boring-stuff/#train-validation-test-split)


Please download the folders *cats*, *dogs* and *wilds* from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/Computer%20Vision/CNN%20for%20Multi%20Class%20Classification) and navigate to the project's root directory in the terminal. The notebook must be started from the location where the three files are stored. 

For this please download the preprocessing_multi_CNN.py file from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/Computer%20Vision/CNN%20for%20Multi%20Class%20Classification) and place this file next to the folders *cats*, *dogs* and *wilds* and start your Jupyter notebook from here.


## 3.1 Train-Validation-Test Split


```r
c_train, d_train, w_train, \
c_val, d_val, w_val, \
c_test, d_test, w_test = Train_Validation_Test_Split('cats', 'dogs', 'wilds')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p1.png)


## 3.2 Obtaining the lists of randomly selected images


```r
list_cats_training = c_train
list_dogs_training = d_train
list_wilds_training = w_train

list_cats_validation = c_val
list_dogs_validation = d_val
list_wilds_validation = w_val

list_cats_test = c_test
list_dogs_test = d_test
list_wilds_test = w_test
```


## 3.3 Determination of the directories



```r
root_directory = os.getcwd()

train_dir = os.path.join(root_directory, 'animals\\train')
validation_dir = os.path.join(root_directory, 'animals\\validation')
test_dir = os.path.join(root_directory, 'animals\\test')
```


## 3.4 Obtain the total number of training, validation and test images



```r
num_cats_img_train = len(list_cats_training)
num_dogs_img_train = len(list_dogs_training)
num_wilds_img_train = len(list_wilds_training)

num_train_images_total = num_cats_img_train + num_dogs_img_train + num_wilds_img_train

print('Total training cat images: ' + str(num_cats_img_train))
print('Total training dog images: ' + str(num_dogs_img_train))
print('Total training wild images: ' + str(num_wilds_img_train))
print()
print('Total training images: ' + str(num_train_images_total))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p2.png)



```r
num_cats_img_validation = len(list_cats_validation)
num_dogs_img_validation = len(list_dogs_validation)
num_wilds_img_validation = len(list_wilds_validation)

num_validation_images_total = num_cats_img_validation + num_dogs_img_validation + num_wilds_img_validation

print('Total validation cat images: ' + str(num_cats_img_validation))
print('Total validation dog images: ' + str(num_dogs_img_validation))
print('Total validation wild images: ' + str(num_wilds_img_validation))
print()
print('Total validation images: ' + str(num_validation_images_total))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p3.png)



```r
num_cats_img_test = len(list_cats_test)
num_dogs_img_test = len(list_dogs_test)
num_wilds_img_test = len(list_wilds_test)

num_test_images_total = num_cats_img_test + num_dogs_img_test + num_wilds_img_test

print('Total test cat images: ' + str(num_cats_img_test))
print('Total test dog images: ' + str(num_dogs_img_test))
print('Total test wild images: ' + str(num_wilds_img_test))
print()
print('Total test images: ' + str(num_test_images_total))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p4.png)



# 4 Feature Extraction with Data Augmentation

## 4.1 Name Definitions



```r
checkpoint_no = 'ckpt_1_CNN_with_TF_VGG19_with_DataAug'
model_name = 'Animals_CNN_TF_VGG19_epoch_30_ES'
```


## 4.2 Parameter Settings



```r
img_height = 150
img_width = 150
input_shape = (img_height, img_width, 3)

n_batch_size = 32

n_steps_per_epoch = int(num_train_images_total / n_batch_size)
n_validation_steps = int(num_validation_images_total / n_batch_size)
n_test_steps = int(num_test_images_total / n_batch_size)

n_epochs = 30

num_classes = len(os.listdir(train_dir))

print('Input Shape: '+'('+str(img_height)+', '+str(img_width)+', ' + str(3)+')')
print('Batch Size: ' + str(n_batch_size))
print()
print('Steps per Epoch: ' + str(n_steps_per_epoch))
print()
print('Validation Steps: ' + str(n_validation_steps))
print('Test Steps: ' + str(n_test_steps))
print()
print('Number of Epochs: ' + str(n_epochs))
print()
print('Number of Classes: ' + str(num_classes))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p5.png)



## 4.3 Instantiating the VGG19 convolutional base


```r
VGG19_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

conv_base = VGG19_base
```


```r
conv_base.summary()
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p6.png)


## 4.4 Instantiating a densely connected classifier

### 4.4.1 Layer Structure


```r
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
```


```r
model.summary()
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p7.png)


```r
print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p8.png)


```r
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
```


```r
model.summary()
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p9.png)


```r
print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p10.png)


### 4.4.2 Configuring the model for training



```r
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```


### 4.4.3 Using ImageDataGenerator



```r
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=n_batch_size,
        class_mode='categorical')


validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_height, img_width),
        batch_size=n_batch_size,
        class_mode='categorical')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p11.png)


## 4.5 Callbacks



```r
# Prepare a directory to store all the checkpoints.
checkpoint_dir = checkpoint_no
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
```


```r
keras_callbacks = [ModelCheckpoint(filepath = checkpoint_dir, 
                                   monitor='val_loss', save_best_only=True, mode='auto'),
                   EarlyStopping(monitor='val_loss', patience=5, mode='auto', 
                                 min_delta = 0, verbose=1)]
```


## 4.6 Fitting the model



```r
history = model.fit(
      train_generator,
      steps_per_epoch=n_steps_per_epoch,
      epochs=n_epochs,
      validation_data=validation_generator,
      validation_steps=n_validation_steps,
      callbacks=keras_callbacks)
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p12.png)



## 4.7 Obtaining the best model values



```r
hist_df = pd.DataFrame(history.history)
hist_df['epoch'] = hist_df.index + 1
cols = list(hist_df.columns)
cols = [cols[-1]] + cols[:-1]
hist_df = hist_df[cols]
hist_df.to_csv(checkpoint_no + '/' + 'history_df_' + model_name + '.csv')
hist_df.head()
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p13.png)



```r
values_of_best_model = hist_df[hist_df.val_loss == hist_df.val_loss.min()]
values_of_best_model
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p14.png)


## 4.8 Obtaining class assignments



```r
class_assignment = train_generator.class_indices

df = pd.DataFrame([class_assignment], columns=class_assignment.keys())
df_stacked = df.stack()
df_temp = pd.DataFrame(df_stacked).reset_index().drop(['level_0'], axis=1)
df_temp.columns = ['Category', 'Allocated Number']
df_temp.to_csv(checkpoint_no + '/' + 'class_assignment_df_' + model_name + '.csv')

print('Class assignment:', str(class_assignment))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p15.png)



## 4.9 Validation



```r
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p16.png)



## 4.10 Load best model



```r
# Loading the automatically saved model
model_reloaded = load_model(checkpoint_no)

# Saving the best model in the correct path and format
root_directory = os.getcwd()
checkpoint_dir = os.path.join(root_directory, checkpoint_no)
model_name_temp = os.path.join(checkpoint_dir, model_name + '.h5')
model_reloaded.save(model_name_temp)

# Deletion of the automatically created folders/.pb file under Model Checkpoint File.
folder_name_temp1 = os.path.join(checkpoint_dir, 'assets')
folder_name_temp2 = os.path.join(checkpoint_dir, 'variables')
file_name_temp = os.path.join(checkpoint_dir, 'saved_model.pb')

shutil.move(file_name_temp, folder_name_temp1)
shutil.rmtree(folder_name_temp1, ignore_errors=True)
shutil.rmtree(folder_name_temp2, ignore_errors=True)
```


```r
best_model = load_model(model_name_temp)
```


## 4.11 Model Testing



```r
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=n_batch_size,
        class_mode='categorical')

test_loss, test_acc = best_model.evaluate(test_generator, steps=n_test_steps)
print()
print('Test Accuracy:', test_acc)
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p17.png)

For later use, I save all necessary metrics separately.


```r
pk.dump(img_height, open(checkpoint_dir+ '\\' +'img_height.pkl', 'wb'))
pk.dump(img_width, open(checkpoint_dir+ '\\' +'img_width.pkl', 'wb'))
```


Our final folder structure now looks like this:

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107s1.png)



## 4.12 Test Out of the Box Pictures



```r
# Determine Checkpoint Dir
checkpoint_dir = 'ckpt_1_CNN_with_TF_VGG19_with_DataAug'

# Load best model
best_model = load_model(checkpoint_dir + '/' + 'Animals_CNN_TF_VGG19_epoch_30_ES.h5')

# Load the categories
df = pd.read_csv(checkpoint_dir + '/' + 'class_assignment_df_Animals_CNN_TF_VGG19_epoch_30_ES.csv')
df = df.sort_values(by='Allocated Number', ascending=True)
CATEGORIES = df['Category'].to_list()


# Load the used image height and width
img_height_reload = pk.load(open(checkpoint_dir + '/' + 'img_height.pkl','rb'))
img_width_reload = pk.load(open(checkpoint_dir + '/' + 'img_width.pkl','rb'))


print('Model Summary :' + str(best_model.summary()))
print()
print()
print('CATEGORIES : ' + str(CATEGORIES))
print()
print('Used image height: ' + str(img_height_reload))
print('Used image width: ' + str(img_width_reload))
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p18.png)



```r
img_pred = cv2.imread('out of the box pic/test_cat_pic_1.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p19.png)



```r
img_pred = cv2.imread('out of the box pic/test_cat_pic_2.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p20.png)



```r
img_pred = cv2.imread('out of the box pic/test_cat_pic_3.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p21.png)



```r
img_pred = cv2.imread('out of the box pic/test_dog_pic_1.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p22.png)



```r
img_pred = cv2.imread('out of the box pic/test_dog_pic_2.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p23.png)



```r
img_pred = cv2.imread('out of the box pic/test_dog_pic_3.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p24.png)



```r
img_pred = cv2.imread('out of the box pic/test_wild_pic_1.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p25.png)



```r
img_pred = cv2.imread('out of the box pic/test_wild_pic_2.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p26.png)



```r
img_pred = cv2.imread('out of the box pic/test_wild_pic_3.jpg')

print(plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)))

img_pred = cv2.resize(img_pred,(img_height_reload,img_width_reload))
img_pred = np.reshape(img_pred,[1,img_height_reload,img_width_reload,3])

classes = np.argmax(best_model.predict(img_pred), axis=-1)

print()
print('------------------------------------')
print('Predicted Class: ' + CATEGORIES[int(classes[0])])
print('------------------------------------')
```

![](/post/2021-01-19-cv-cnn-with-transfer-learning-for-multi-label-classification_files/p107p27.png)



# 5 Conclusion

In addition to my post [Computer Vision - CNN with Transfer Learning](https://michael-fuchs-python.netlify.app/2021/01/17/computer-vision-cnn-with-transfer-learning/) I have shown here how to use pre-trained neural networks to solve multiple classification problems. 


# 6 Link to the GitHub Repository

Here is the link to my GitHub repository where I have listed all necessary steps: [CV-CNN-with-Transfer-Learning-for-Multi-Class-Classification](https://github.com/MFuchs1989/CV-CNN-with-Transfer-Learning-for-Multi-Class-Classification)



**References**

The content of the entire post was created using the following sources:

Chollet, F. (2018). Deep learning with Python (Vol. 361). New York: Manning.


