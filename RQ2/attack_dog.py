from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
import numpy as np
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    res = pickle.load(f)
    data = res[0]
    label = res[1]
    return data, label


def get_data():

    all_data = []
    all_label = []
    all_path = ['base_data.pkl', 'augmentation_flip_horizontal_vertical.pkl', 'augmentation_random_brightness.pkl',
                'augmentation_random_rotation.pkl', 'augmentation_random_translation.pkl', 'augmentation_random_zoom.pkl']

    for path in all_path:
        data, label = read_data(path)
        all_data = all_data + data
        all_label = all_label + label

    for i in range(len(all_data)):
        tmp_x = all_data[i]
        if np.max(tmp_x) > 5:
            tmp_x = tmp_x * 1.0 / 255
            all_data[i] = tmp_x

    dic_key_value = {'Miniature schnauzer': 0, 'French bulldog': 1, 'Cocker spaniel': 2, 'Irish wolfhound': 3, 'Norwich terrier': 4,
                     'Gordon setter': 5, 'Bernese mountain dog': 6, 'Great dane': 7, 'Papillon': 8, 'Norwegian elkhound': 9}

    all_data = np.array(all_data)

    numeric_label = np.array([dic_key_value[i] for i in all_label])

    x_train, x_test, y_train, y_test = train_test_split(all_data, numeric_label, test_size=0.2, random_state=5)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data()


def cnn(x_train, num_classes):

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


model = cnn(x_train, 10)
classifier = get_train_model(model, x_train, y_train, 64, 10, 'cnn.h5')


def fgm_x_adv(classifier):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)  # eps 值越大，攻击越大，一般eps=0.1
    x_test_adv = attack.generate(x=x_test)
    output = open('fsgm_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.generate(x=x_train)
    output = open('fsgm_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


def patch_x_adv(classifier):
    attack = AdversarialPatch(classifier=classifier)
    x_test_adv = attack.apply_patch(x=x_test, scale=0.3)
    output = open('patch_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.apply_patch(x=x_train, scale=0.3)
    output = open('patch_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


def bim_x_adv(classifier):
    attack = BasicIterativeMethod(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    output = open('bim_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.generate(x=x_train)
    output = open('bim_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


def pgd_x_adv(classifier):
    attack = ProjectedGradientDescent(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    output = open('pgd_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.generate(x=x_train)
    output = open('pgd_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


def newf_x_adv(classifier):
    attack = NewtonFool(classifier=classifier)
    x_test_adv = attack.generate(x=x_test)
    output = open('newf_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.generate(x=x_train)
    output = open('newf_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


def sa_x_adv(classifier):
    attack = SquareAttack(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    output = open('sa_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output, protocol=4)
    x_train_adv = attack.generate(x=x_train)
    output = open('sa_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output, protocol=4)


if __name__ == '__main__':
    fgm_x_adv(classifier)
    patch_x_adv(classifier)
    bim_x_adv(classifier)
    pgd_x_adv(classifier)
    newf_x_adv(classifier)
    sa_x_adv(classifier)




