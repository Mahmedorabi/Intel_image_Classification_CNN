# Intel Image Classification
This notebook demonstrates an image classification task using TensorFlow and Keras. It focuses on classifying images into six categories: buildings, forest, glacier, mountain, sea, and street.

## Requirements
The following libraries are required to run this notebook:

- numpy
- os
- pathlib
- matplotlib
- tensorflow
- You can install them using pip:

```bash
pip install numpy matplotlib tensorflow
```
## Data
The dataset consists of images categorized into six classes. The paths for the training and testing data are specified as follows:

```python
# Change file location if you wanna use it
train_path = "D:/Project AI/intel/seg_train/seg_train"
test_path = "D:/Project AI/intel/seg_test/seg_test"
```
## Data Visualization


![output visual of intel](https://github.com/Mahmedorabi/Intel_image_Classification/assets/105740465/eec8099c-c5ba-4093-8ef4-d75f44f483e0)



## Data Preparation
Data augmentation and preparation are done using ImageDataGenerator:

```python

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator=ImageDataGenerator(rescale=1/255)

rain_data=data_generator.flow_from_directory(train_path,
                                             target_size=(150,150),
                                             batch_size=128,
                                             shuffle=True)

test_data=data_generator.flow_from_directory(test_path,
                                            target_size=(150,150),
                                            batch_size=32,
                                            shuffle=True)
```
## Model Creation
The model is created using the Sequential API from Keras:

```python

model=Sequential()

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[150,150,3]))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.20))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.30))

model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.40))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))

model.add(Dense(6,activation='softmax'))
```
## Training
The model is compiled and trained with the following settings:

```python
# Model compile
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# Model Fitting
model_hist=model.fit(train_data,validation_data=test_data,epochs=10)
```
## Evaluation
The model’s performance is evaluated on the test set:

```python
# Evaluat of training
loss,acc=model.evaluate(train_data)
print(f'Accuracy of Training data is: {acc*100}')

# Evaluat of testing
loss,acc=model.evaluate(test_data)
print(f'Accuracy of Testing data is: {acc*100}')


```
## Visualization of model performance
Training and validation accuracy and loss are visualized using Matplotlib:

```python
fig,ax = plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Model Preformance Visualization",fontsize=20)
ax[0].plot(model_hist.history['loss'],label='Training Loss')
ax[0].plot(model_hist.history['val_loss'],label='Testing Loss')
ax[0].set_title('Training Loss VS. Testing Loss')
ax[0].legend()

ax[1].plot(model_hist.history['accuracy'],label='Training Accuracy')
ax[1].plot(model_hist.history['val_accuracy'],label='Testing Accuracy')
ax[1].set_title('Training Accuracy VS. Testing Accuracy')
ax[1].legend()
plt.show()
```

![output model prefomance of intel](https://github.com/Mahmedorabi/Intel_image_Classification/assets/105740465/1566fd38-e62c-4b59-b998-11fa4a6dfe30)

## Predication new Image




![output predict intel](https://github.com/Mahmedorabi/Intel_image_Classification/assets/105740465/1417df57-b3ec-45f1-bf87-044112b76f37)



![output predict intel1](https://github.com/Mahmedorabi/Intel_image_Classification/assets/105740465/92b3cbae-44dc-4e00-a631-2ddd8c935b1f)














