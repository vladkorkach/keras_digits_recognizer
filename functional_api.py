# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization, Conv2D, MaxPool2D, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class MNISTmodel:
    seed = 42
    epochs = None
    batch_size = None
    input_shape = None
    number_of_classes = None
    cval_test_size = 0.1
    
    model = None
    
    def __init__(self, train, test, **kwargs):
        if not (isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)):
            train = pd.read_csv(train)
            test = pd.read_csv(test)

        self.input_shape = kwargs.get('input_shape') or tuple([28, 28, 1])
        self.batch_size = kwargs.get('batch_size') or 64
        self.epochs = kwargs.get('epochs') or 20
        self.number_of_classes = kwargs.get('number_of_classes') or 10

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
        print(self.input_shape)
        x_train = x_train.reshape(number_of_train, *self.input_shape) # (n, 28, 28, 1)
        self.x_test  = x_test.reshape(number_of_test, *self.input_shape) # (n, 28, 28, 1)
        
        # cross validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_train, y_train, 
                                                                              test_size=self.cval_test_size,
                                                                              random_state=self.seed)

    def onehot_encode(self, labels):
        return to_categorical(labels, self.number_of_classes)
    
    def build(self):
        inputs = Input(shape=self.input_shape)

        c1   = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(inputs)
        c2   = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(c1)
        p3   = MaxPool2D(pool_size=(2, 2))(c2)

        dr4  = Dropout(0.2)(p3)

        c5   = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(dr4)
        c6   = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)
        p7   = MaxPool2D(pool_size=(2, 2))(c6)

        dr8  = Dropout(0.25)(p7)

        c9   = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(dr8)

        dr10 = Dropout(0.25)(c9)

        fl11 = Flatten()(dr10)

        d12  = Dense(128, activation='relu')(fl11)

        bn13 = BatchNormalization()(d12)
        
        dr14 = Dropout(0.25)(bn13)

        d15  = Dense(self.number_of_classes, activation='softmax')(dr14)

        self.model = Model(inputs, outputs=d15)

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
