import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D # type: ignore
from tensorflow.python.keras import activations

from deeplearning_models import functional_model, MyCustomModel
from my_utils import display_some_examples

# This Practice Project uses Python 3.8, then follow normal tensorflow instructions for installation < use in VENV using miniconda within VSCode. Use CMD terminal not powershell, etc

# Multiple approches for Neural Networks

# 1st Approch: tensorflow.keras.Sequential
seq_model = tf.keras.Sequential(
    [
        #28x28 & 1 color channel
        Input(shape=(28, 28, 1)), 
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(32, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        
        GlobalAvgPool2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation= 'softmax')
    ]
)


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("x_train.shape = ", y_train.shape)
    print("x_train.shape = ", x_test.shape)
    print("x_train.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    #turn to float 32 because values between 0 - 255, when you divide it becomes 0 so has to turn to float
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print("Changed")
    print("x_train.shape = ", x_train.shape)
    print("x_train.shape = ", y_train.shape)
    print("x_train.shape = ", x_test.shape)
    print("x_train.shape = ", y_test.shape)

    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    # 100 images     EXAMPLE
    # prediction: 88 correct, 12 incorrect
    # 88% accurate

    # Model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2) # Validation_split=0.2, 20% is used for validation while 80% is used for training

    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)


