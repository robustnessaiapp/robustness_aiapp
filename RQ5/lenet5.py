import tensorflow as tf

from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD


def Lenet5():
    nb_classes = 10
    kernel_size = (5, 5)
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='dense_1')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':

    batch_size = 64
    epochs = 30
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    model = Lenet5()
    model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, validation_data=(x_test, y_test), verbose=1)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save('lenet5.h5')
    print(model.summary())
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.post_training_quantize = True
    tflite_model = converter.convert()
    open('lenet5.tflite', "wb").write(tflite_model)