# import the necessary packages
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.convolutional import MaxPooling2D
from tensorflow_core.python.keras.layers.core import Activation
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.core import Flatten
from tensorflow.keras.models import Sequential


class networkArchFonc:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(16, (2, 2), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (2, 2), padding="same"))  # kernelere göre conv yerni bir matris oluşturma
        model.add(Activation("relu"))  # reulu: negatif değerleri çevirme relu sıfıra elu e üzeri
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())  # düzleştirme ?
        model.add(Dense(500))  # fully connected
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #	print(model.summary())
        # return the constructed network architecture
        return model
