from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD


batch_size = 64
nb_epochs = 80
num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def cnn():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, name='dense_1'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def get_train_model(model, x_train, y_train, batch_size, nb_epochs, save_model_path):
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=True)
    model.save(save_model_path)
    return classifier


model = cnn()
classifier = get_train_model(model, x_train, y_train, batch_size, nb_epochs, 'cnn_300.h5')


def fsgm_x_adv(classifier):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('fsgm_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('fsgm_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def patch_x_adv(classifier):
    attack = AdversarialPatch(classifier=classifier)
    x_test_adv = attack.apply_patch(x=x_test, scale=0.3)
    x_train_adv = attack.apply_patch(x=x_train, scale=0.3)
    output = open('patch_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('patch_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def bim_x_adv(classifier):
    attack = BasicIterativeMethod(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('bim_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('bim_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def pgd_x_adv(classifier):
    attack = ProjectedGradientDescent(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('pgd_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('pgd_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def newf_x_adv(classifier):
    attack = NewtonFool(classifier=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('newf_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('newf_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def sa_x_adv(classifier):
    attack = SquareAttack(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('sa_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('sa_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


if __name__ == '__main__':
    fsgm_x_adv(classifier)
    patch_x_adv(classifier)
    bim_x_adv(classifier)
    pgd_x_adv(classifier)
    newf_x_adv(classifier)
    sa_x_adv(classifier)


