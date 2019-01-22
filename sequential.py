# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization, Conv2D, MaxPool2D, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# hyperparams
input_shape = (28, 28, 1)
batch_size = 16
epochs = 4
number_of_classes = 10

class MNISTmodel:
    seed = 42
    epochs = epochs
    batch_size = batch_size
    input_shape = input_shape
    number_of_classes = number_of_classes
    cval_test_size = 0.1
    
    model = None
    
    def __init__(self, train, test):
        if not (isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)):
            train = pd.read_csv(train)
            test = pd.read_csv(test)

        self.preprocessing(train, test)

    
    def preprocessing(self, train, test):
        # only pixels to numpy ndarray
        x_train = train.iloc[:, 1:].values.astype('float32')
        x_test  = test.values.astype('float32')

        # train labels for X_train
        y_train = train.iloc[:, 0].values.astype('float32')
        y_train = self.onehot_encode(y_train)

        # normalize pixels => [0..1]
        x_train = x_train / 255.
        x_test  = x_test  / 255.

        # reshape
        number_of_train = x_train.shape[0]
        number_of_test  = x_test.shape[0]
        
        # assign
        x_train = x_train.reshape(number_of_train, *self.input_shape) # (n, 28, 28, 1)
        self.x_test  = x_test.reshape(number_of_test, *self.input_shape) # (n, 28, 28, 1)
        
        # cross validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, 
                                                                              test_size=self.cval_test_size,
                                                                              random_state=self.seed)

    def onehot_encode(self, labels):
        return to_categorical(labels, self.number_of_classes)
    
    def build_sequential(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=self.input_shape))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(self.number_of_classes, activation='softmax'))

    def compile(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=RMSprop(),
                   metrics=['accuracy'])
        
        self.lerning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=3,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.0001)
    
    @staticmethod
    def datagen():
        return ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
    
    def summary(self):
        self.model.summary()
        
    def evaluate(self, x=None, y=None, verbose=0):
        final_loss, final_acc = None, None
        
        if x is None and y is None:
            x, y = self.x_val, self.y_val
            
        final_loss, final_acc = self.model.evaluate(x, y, verbose=verbose)
        print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
        
    def predict(self, test=None, classes=False):
        if test is None:
            test = self.x_test

        if classes:
            return self.model.predict_classes(test)
        else:
            return self.model.predict(test)
        
    
from subprocess import check_output
print(check_output(['ls', './input']).decode('utf8'))

train = pd.read_csv('./input/train.csv')
test  = pd.read_csv('./input/test.csv')

mnist = MNISTmodel(train, test)
mnist.build_sequential()
mnist.compile()
mnist.summary()

datagen = MNISTmodel.datagen()
datagen.fit(mnist.x_train)

history = mnist.model.fit_generator(datagen.flow(mnist.x_train, mnist.y_train, batch_size=mnist.batch_size),
                                  epochs=mnist.epochs,
                                  verbose=1,
                                  validation_data=(mnist.x_val, mnist.y_val),
                                  steps_per_epoch=mnist.x_train.shape[0] // mnist.batch_size,
                                  callbacks=[mnist.lerning_rate_reduction])

mnist.evaluate()

y_pred = mnist.predict(mnist.x_test, classes=True)
# y_pred
# len(y_pred)

# for i in range(0, 6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(mnist.x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
#     plt.title(y_pred[i]);

df = pd.DataFrame({ 'ImageId': list(range(1, len(y_pred) + 1)), 'Label': y_pred })
df.to_csv('result_keras_implementation_2.csv', sep=',', index=False, index_label='ImageId')