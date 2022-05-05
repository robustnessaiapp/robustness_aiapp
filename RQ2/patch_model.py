from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pickle



def patch_model():
    input = Input(shape=X_train.shape[1:], dtype='float32', name='main_input')
    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    patch_model_out = Dense(10, name='dense0')(x)

    mobile_input = Input(shape=(127,), name='ai_app_prediction_vector')  # Vector of AI App embedded model

    x = tf.keras.layers.concatenate([mobile_input, patch_model_out])
    x = Dense(10, name='dense_1')(x)
    x = Activation('softmax')(x)

    model = Model(inputs=[mobile_input, input], outputs=x)

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    res = pickle.load(f)
    data = res[0]
    label = res[1]
    return data, label


def read_data_attack(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def read_data_pre(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_augmentation_data():
    all_data = []  # [img0, img1, ...] <255
    all_label = []  # ['French bulldog', 'French bulldog', 'French bulldog', ...]
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

    all_data = np.array(all_data)

    dic_key_value = {'Miniature schnauzer': 0, 'French bulldog': 1, 'Cocker spaniel': 2, 'Irish wolfhound': 3,
                     'Norwich terrier': 4, 'Gordon setter': 5, 'Bernese mountain dog': 6, 'Great dane': 7,
                     'Papillon': 8, 'Norwegian elkhound': 9}

    numeric_label = np.array([dic_key_value[i] for i in all_label])
    all_label = tf.keras.utils.to_categorical(numeric_label, 10)
    all_label = all_label.astype('float32')
    x_train, x_test, y_train, y_test = train_test_split(all_data, numeric_label, test_size=0.2, random_state=5)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    attack_label = y_test

    return all_data, all_label, attack_label


def get_attack_data():
    bim_data = read_data_attack('bim_x_adv.pkl')
    fsgm_data = read_data_attack('fgm_x_adv.pkl')
    newf_data = read_data_attack('newf_x_adv.pkl')
    patch_data = read_data_attack('patch_x_adv.pkl')
    pgd_data = read_data_attack('pgd_x_adv.pkl')
    sa_data = read_data_attack('sa_x_adv.pkl')
    return bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data


if __name__ == '__main__':

    print('===============start to read data=====================')
    all_data, all_label, attack_label = get_augmentation_data()
    print('====================================')
    all_x_train, all_x_test, all_y_train, all_y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=5)
    print('====================================')
    bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data = get_attack_data()
    print('====================================')
    bim_x_train, bim_x_test, bim_y_train, bim_y_test = train_test_split(bim_data, attack_label, test_size=0.2, random_state=5)
    print('====================================')
    fsgm_x_train, fsgm_x_test, fsgm_y_train, fsgm_y_test = train_test_split(fsgm_data, attack_label, test_size=0.2, random_state=5)
    print('====================================')
    newf_x_train, newf_x_test, newf_y_train, newf_y_test = train_test_split(newf_data, attack_label, test_size=0.2, random_state=5)
    print('====================================')
    patch_x_train, patch_x_test, patch_y_train, patch_y_test = train_test_split(patch_data, attack_label, test_size=0.2, random_state=5)
    print('====================================')
    pgd_x_train, pgd_x_test, pgd_y_train, pgd_y_test = train_test_split(pgd_data, attack_label, test_size=0.2, random_state=5)
    print('====================================')
    sa_x_train, sa_x_test, sa_y_train, sa_y_test = train_test_split(sa_data, attack_label, test_size=0.2, random_state=5)

    bim_x_test_pre = read_data_pre('bim_x_test_pre.pkl')
    bim_x_train_pre = read_data_pre('bim_x_train_pre.pkl')
    fsgm_x_test_pre = read_data_pre('fgm_x_test_pre.pkl')
    fsgm_x_train_pre = read_data_pre('fgm_x_train_pre.pkl')
    newf_x_test_pre = read_data_pre('newf_x_test_pre.pkl')
    newf_x_train_pre = read_data_pre('newf_x_train_pre.pkl')
    patch_x_test_pre = read_data_pre('patch_x_test_pre.pkl')
    patch_x_train_pre = read_data_pre('patch_x_train_pre.pkl')
    pgd_x_test_pre = read_data_pre('pgd_x_test_pre.pkl')
    pgd_x_train_pre = read_data_pre('pgd_x_train_pre.pkl')
    sa_x_test_pre = read_data_pre('sa_x_test_pre.pkl')
    sa_x_train_pre = read_data_pre('sa_x_train_pre.pkl')

    all_data_pre = read_data_pre('all_data_pre.pkl')

    X_train = np.concatenate((all_data, bim_x_train, fsgm_x_train, newf_x_train, patch_x_train, pgd_x_train, sa_x_train), axis=0)
    X_test = np.concatenate((bim_x_test, fsgm_x_test, newf_x_test, patch_x_test, pgd_x_test, sa_x_test), axis=0)
    Y_train = np.concatenate((all_label, bim_y_train, fsgm_y_train, newf_y_train, patch_y_train, pgd_y_train, sa_y_train), axis=0)
    Y_test = np.concatenate((bim_y_test, fsgm_y_test, newf_y_test, patch_y_test, pgd_y_test, sa_y_test), axis=0)

    X_train_pre = np.concatenate((all_data_pre, bim_x_train_pre, fsgm_x_train_pre, newf_x_train_pre, patch_x_train_pre, pgd_x_train_pre, sa_x_train_pre), axis=0)
    X_test_pre = np.concatenate((bim_x_test_pre, fsgm_x_test_pre, newf_x_test_pre, patch_x_test_pre, pgd_x_test_pre, sa_x_test_pre), axis=0)

    print('==============start train======================')
    model = patch_model()
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_10.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_20.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_30.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_40.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_50.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_60.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_70.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_80.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_90.h5')
    model.fit([X_train_pre, X_train], Y_train, validation_data=([X_test_pre, X_test], Y_test), shuffle=True, epochs=10, batch_size=64)
    model.save('patch_model_100.h5')
