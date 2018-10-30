## <center>GR5293 Group Project Report</center>

## <center>Three Billboards</center>

**Huiwen Luo (hl3111), Rui Zhang (rz2406), Lujia Wang (lw2772)**


### CNN Introduction

Convolutional neural network is used for image recognition. The convolution step is used to  extract features from the input images. We slide the filter matrix over the original image and on each position do element wise multiplication of the two matrices. Then we add up the answers to get an single element of the output matrix. In the pooling step, large images are shrunk down but the most important information is preserved. Max pooling, a most widely used pooling method, is basically stepping a window across an image and catching the largest number in the window in each step. A pooling layer will give same number of output images as before but with fewer pixels, which also help reduce computational load. After we get the output images containing high level features from applying convolution layer and pooling layer, we use dense layer to determine those features we extracted are most correlated to a particular class. The output will be a vector indicating which class the image should be in.



### CNN Parameters  

- Global Parameters:

| Parameter Name | Usage                                                  | Value |
| :------------- | :----------------------------------------------------- | :---: |
| epoch          | An epoch is an iteration over the entire x and y data. |  30   |
| num_classes    | Dimensionality of the output space                     |  10   |

- Conv2D:

| Parameter Name | Usage                                                        |    Value     |
| :------------- | :----------------------------------------------------------- | :----------: |
| kernel_size    | Length of the 2D convolution window                          | (5,5)  (3,3) |
| strides        | Stride length of the convolution                             |    (1,1)     |
| activation     | The activation function to use                               |    'relu'    |
| padding        | One of “valid”, “causal” or “same”. “same” will lead to an output that has the same length as the input; “causal” will give a causal convolution; and “valid” means no padding. |    'same'    |

- MaxPooling2D


| Parameter Name | Usage                                                        | Value |
| :------------- | :----------------------------------------------------------- | :---: |
| pool_size      | Size of the max pooling windows, factors by which to downscale (vertical, horizontal). | (2,2) |

- Dropout

| Parameter Name | Usage                           |  Value  |
| :------------- | :------------------------------ | :-----: |
| rate           | Fraction of input units to drop | 0.250.5 |

- Dense

| Parameter Name | Usage                                                        |      Value       |
| :------------- | :----------------------------------------------------------- | :--------------: |
| units          | Dimensionality of the output space                           |       1024       |
| activation     | Activation function to use: 'relu' for the first Dense layer, 'softmax' for the logits layer. | 'relu' /'softmax |

- ImageDataGenerator

| Parameter Name     | Usage                                                        | Value |
| :----------------- | :----------------------------------------------------------- | :---: |
| rotation_range     | Degree range of random rotations                             |  10   |
| width_shift_range  | Shift range horizontally                                     |  0.1  |
| height_shift_range | Shift range vertically                                       |  0.1  |
| shear_range        | Shear Intensity (shear angle in counter-clockwise direction in degrees) |  0.1  |
| zoom_range         | Range for random zoom                                        |  0.1  |
| batch_size         | Size of batches of augmented data in the *.flow* method      |  64   |

- ReduceLROnPlateau:


| Parameter Name | Usage                                                        |   Value   |
| :------------- | :----------------------------------------------------------- | :-------: |
| monitor        | Quantity to be monitored                                     | 'val_acc' |
| patience       | Number of epochs with no improvement after which learning rate will be reduced |     3     |
| verbose        | 0 means quiet and 1 means updating messages                  |     1     |
| factor         | Factor by which the learning vector will be reduced.         |    0.5    |
| min_lr         | Lower bound on the learning rate                             |  0.0001   |



### Code Logic

- #### Data Preparation

First we set some global variables, number of classes, input image dimensions and epochs to train. We then load the MNIST data sets, normalize and reshape the data sets, and convert labels to binary class metrics.

```python
num_classes = 10
epochs = 30
# input image dimensions
img_rows, img_cols = 28, 28
# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# normalize data to [0,...,1] rather than original [0,..., 255]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes) #(60000,10)
y_test = keras.utils.to_categorical(y_test, num_classes) #(10000,10)
```

Then, we split the default training set into training and validation set. Thus we get 54000 training samples, 6000 validation samples and 10000 test samples.

```python
# split x train into train and val
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size = 0.1,
                                                  random_state = 1)
```

- #### CNN Model Design

We use Keras Sequential API to build the CNN model. Our model archetechture is **Input -> 2 * (Conv2D(32, 5x5, ReLU) -> MaxPooling2D(2x2) -> Drop out(0.25) -> 2 * (Conv2D(64, 5x5, ReLU)) -> MaxPooling2D(2x2) -> Drop out(0.25) -> Flatten -> Dense(1024) -> Drop out(0.5) -> Dense(10) -> Output**.  This archetechture is inspired by the VGGNet developed by Simonyan and Zisserman in 2014.

A summary is presented below.

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        832       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 32)        25632     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 64)        51264     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 64)        102464    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              3212288   
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                10250     
=================================================================
Total params: 3,402,730
Trainable params: 3,402,730
Non-trainable params: 0
_________________________________________________________________
```

For the first two convolutional layers, we apply 32 filters to convolve with the inputs. The inputs shape is [28, 28, 1]. The kernel size is 5 $\times$ 5, which are the dimensions of the filters. In order to specify that the outputs should have the same height and width values as the inputs, we set `padding = 'same'` here. This allows Keras to add 0 values to the edges of the inputs to preserve height and width of 28. Then we use ReLU activation to apply to the outputs of the convolution. The outputs shape after convolution is [28, 28, 32].

```python
model = Sequential()
# convolutional layer #1
model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same',
                 activation = 'relu', input_shape = input_shape))
# convolutional layer #2
model.add(Conv2D(32, kernel_size = (5, 5), padding = 'same',
                 activation = 'relu')) 

model.add(Dropout(0.25))
```

Next, we connect our first pooling layer to the convolutional layers we just created. We use `MaxPooling2D()` to construct a layer that performs max pooling with 2 $\times$ 2 filter and stride of 2 (default value is equal to the `pool_size`). This step extracts the maximum value in each non-overlapping 2 $\times$ 2 block of the inputs, which reduces the inputs' height and width by 50%. Thus, the outputs shape is [14, 14, 32].

```python
# Pooling layer #1
model.add(MaxPooling2D(pool_size = (2, 2)))
```

After combining the convolution layers and pooling layer, we add the `Dropout` to prevent our model from overfitting. It is a regularization method, where a proportion of nodes are randomly dropped out (setting their weights to zero) for each training sample. We set the drop out rate to 0.25, which means 25% of the nodes will be ignored.

```python
model.add(Dropout(0.25))
```

We then repeat previous three steps and add more layers to our model. This time we set the filter size to be 64. The rest parameters for convolution layers, pooling layers and drop out step are the same. Now, the outputs shape is [7, 7, 64] .

```python
# Convolution layer #3
model.add(Conv2D(64, kernel_size = (5, 5), padding = 'same',
                 activation='relu'))
# Convolution layer #4
model.add(Conv2D(64, kernel_size = (5, 5), padding = 'same',
                 activation='relu'))
# Pooling layer #2
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
```

Before we add the a dense layer, we need to flatten out feature map to `[batch_size, features]` , so that the outputs size is [`batch_size` , 7 * 7 * 64], which is [`batch_size` , 3136].  This step is used to combine all the found local features of the previous convolutional/pooling layers.

```python
model.add(Flatten())
```

Next, we want to add a dense layer (with 1024 neurons and ReLU activation) to our CNN to perform classification on the features extracted by the convolution/pooling layers. Then, we again use a `Dropout` with rate of 0.5 to improve generalization and reduces overfitting.

```python
# Dense layer #1
model.add(Dense(1024, activation = 'relu'))
# Drop out
model.add(Dropout(0.5))
```

The final layer in our neural network is the logits layer (with 10 neurons and Softmax activation), which will return the predicted class of each sample.

```python
# Dense layer #2
model.add(Dense(num_classes, activation = 'softmax'))
```

In the end, we compile the model with appropriate optimizer, loss function and metric function.

```python
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
```

A very serious problem of CNN is that the model tends to be overfit very easily. So, we want expand our training data set to avoid overfitting. What we do is to randomly make small alterations to the training samples. In other words, we want to reproduce some variations occuring when someone is writing a digit. Here we use `ImageDataGenerator()` function. The `flow` method generates batches of augmented data. For validation, we don't apply data augmentation to better evaluate the performance of our training model, but still we generates batches of validation set in accordance with the training set.

```python
# randomly rotate some images 10 degrees
# randomly shift some images horizontally 10% of the width
# randomly shift some images vertically 10% of the height
# randomly zoom some images 10%
# randomly shear intensity 10%
gen = ImageDataGenerator(rotation_range=10, 
                         width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         shear_range = 0.1,
                         zoom_range=0.1)

train_generator = gen.flow(x_train, y_train, batch_size=64)
# don't apply data augmentation on validation set
val_gen = ImageDataGenerator()
val_generator = val_gen.flow(x_val, y_val, batch_size=64)
```

In addition, we use a `ReduceLRonPlateau()` function to reduce the learning rate by half if the validation accuracy is not imporoved after 3 epochs.

```python
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

Now, we can fit our model on the training dataset.

```python
mymodel = model.fit_generator(train_generator,
                              steps_per_epoch=60000//64,epochs=epochs,
                              validation_data = val_generator,
                              callbacks=[learning_rate_reduction])
```

After fitting the model, we can evaluate our model on the test dataset.

```python
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

We can plot the changes of loss and accuracy rate over epochs.

```python
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(mymodel.history['loss'], color='b', label="Training loss")
ax[0].plot(mymodel.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(mymodel.history['acc'], color='b', label="Training accuracy")
ax[1].plot(mymodel.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
```

![oss&accurac](/Users/mandy/Downloads/GR5293/group_proj/loss&accuracy.png)



### Improvement Process

We build our starting CNN model from Keras tutorial. The model architecture is **Input -> Conv2D(32, 3x3, ReLU) -> Con2D(64, 3x3, ReLU) -> MaxPooling2D(2x2) -> Drop out(0.25) -> Flatten -> Dense(128) -> Drop out(0.5) -> Dense(10) -> Output**. The test accuracy was ~99.2% after 12 epochs. The sample code has several drawbacks. First, it uses test data as validation data, which is involved in the training process. The consequence is that the test accuracy will be biased and usually higher than the true accuracy rate. Another drawback is the model's architecture tends to be overfitting th training data. The training accuracy can be much higher than the test accuracy.

We then modify the sample model with a much deeper neural network, and split a set of validation samples from the training samples. The validation set is 10% of the training set, which are 6000 samples. Our modified model looks like *'double'* the original sample CNN model. Actually, this model is inspired by the VGGNet model, which has a very clean and organized model structure. The new model architechture is **Input -> 2 * (Conv2D(32, 5x5, ReLU) -> MaxPooling2D(2x2) -> Drop out(0.25) -> 2 * (Conv2D(64, 5x5, ReLU)) -> MaxPooling2D(2x2) -> Drop out(0.25) -> Flatten -> Dense(1024) -> Drop out(0.5) -> Dense(10) -> Output**. This time we reach a test accuracy of 99.47% after 30 epochs. This model improves the test accuracy but still overfits easily. The training accuracy reaches ~99.8% after some epochs. The training time is also longer than the sprevious model because of deeper network and larger number of epochs.

We then want to control the problem of overfitting. In our search, we find a data augmentation approach. It aims to make small alterations to the training handwritten digits images and reproduce the variances during handwritting. This can help prevent the model from overfitting. We also add a learning rate annealer. Its function is to reduce the learning rate by some factor whenever the validation error stops moving after some epochs. With a high learning rate, the system contains too much kinetic energy and the parameter vector bounces around chaotically. With a low learning rate the training is more reliable but optimization will take a lot of time because steps towards the minimum of the loss function are tiny. So we need a higher learning rate at the beginning of the training process and decreases the learning rate as the training goes deeper. In our practice, the learning rate reduces to 0.5 after 10 epochs, 0.25 after 23 epochs, 0.125 after 26 epochs and 0.0625 after 29 epochs. This time, we reach a test accuracy of ~99.6%.  In the plot of training and validation loss/accuracy over epochs, we can see that training accuracy is almost always lower than the validation accuracy. This occurs because we apply data augmentation to the training set, thus it is much more difficult for the model to correctly predict the training label than validation label.

Another problem we need to fix is the running time. The modified model runs more than 7 hrs on a MacBook Pro. We dicide to use *TensorFlow-GPU* backened for Keras. This time, the running time is  less than 5 mins on a GTX1080 GPU. The test accuracy is 99.68%, similar as before, but the computation is much faster.



### Difficulties

Firstly, we have been always dealing with the overfitting problem: we got many train errors very close to 100%, while the test error always below 99.5% even if we tried some of the techniques such as the dropout function to reduce this problem. Finally, we implemented “data augmentation” which has the best results in terms of treating the overfitting problem. It is basically a technique by which we increase the amount of training data using only the information in the training data. 

Secondly, when modifying our model, since we tried a more and more complex process than the original one (i.e. we “double” the layer as well as implementing data augmentation) and set more epochs, we confronted an extremely long running time more than 7 hours. This led to a slow progress of modifying models because it always took long time to check the test error of the model on each stage. We solved this by using TensorFlow GPU, by which we achieve a faster running time to less than 5 minutes. 



### Future Work

- The parameters (eg. filter size, batch size, learnng rate fraction,…) for our current model are fixed. The model fits well for this data set, but may not be very adaptive for future data sets. Before the training, we can use a subset of data, and cross validate the optimal parameters. Then we use the optimal parameters to build CNN model and train the whole data set. 


- Use `BatchNormalization()` or other methods to make computation faster.
- For now, we got a comparatively high prediction rate because our data is large. Sometimes, we do not have enough amount of dataset but still need to build the model and make predictions. We need to create a model that can not only do with large dataset but also smaller dataset.







### Reference

*Keras: The Python Deep Learning library. (n.d.). Retrieved April 28, 2018, from https://keras.io/*

*Keras-team/keras. Retrieved April 29, 2018, from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py*

*Introduction to CNN Keras - 0.997 (top 6%) | Kaggle. (n.d.). Retrieved April 29, 2018, from https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6*

*Spark, C. (n.d.). Deep learning for complete beginners: Convolutional neural networks with keras. Retrieved April 29, 2018, from https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html*
