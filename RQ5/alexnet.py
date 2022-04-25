from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
image_shape = (28, 28, 1)


def AlexNet():
    AlexNet_model = Sequential()
    # 1st convolition layer
    AlexNet_model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size=(11, 11), strides=(4, 4), padding='same'))
    AlexNet_model.add(Activation('relu'))
    AlexNet_model.add(BatchNormalization())
    # Max Pooling layer
    AlexNet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 2nd Conv layer
    AlexNet_model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    AlexNet_model.add(Activation('relu'))
    AlexNet_model.add(BatchNormalization())
    # Max pooling layer
    AlexNet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd Conv layer
    AlexNet_model.add(Conv2D(filters=384, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    AlexNet_model.add(Activation('relu'))
    AlexNet_model.add(BatchNormalization())

    # 4th Conv layer
    AlexNet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet_model.add(Activation('relu'))
    AlexNet_model.add(BatchNormalization())

    # 5th Conv layer
    AlexNet_model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    AlexNet_model.add(Activation('relu'))
    AlexNet_model.add(BatchNormalization())

    # Max pooling layer
    AlexNet_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Flatten layer
    AlexNet_model.add(Flatten())
    # 6th fully connected layer
    AlexNet_model.add(Dense(4096, input_shape=image_shape, activation='relu', name='dense0'))
    AlexNet_model.add(Dropout(0.4))

    # 7th fully connected layer
    AlexNet_model.add(Dense(4096, activation='relu', name='dense1'))
    AlexNet_model.add(Dropout(0.4))

    # 8th fully connected layer
    AlexNet_model.add(Dense(1000, activation='relu', name='dense2'))
    AlexNet_model.add(Dropout(0.4))

    # Output layer
    AlexNet_model.add(Dense(10, activation='softmax', name='dense_1'))
    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    AlexNet_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return AlexNet_model

model = AlexNet()

model.fit(x_train, y_train, batch_size=64, shuffle=True, epochs=30, validation_data=(x_test, y_test),verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
model.save('alexnet.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.post_training_quantize = True
# tflite_model = converter.convert()
# open('alexnet.tflite', "wb").write(tflite_model)
