import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
data_path = 'dog/'


def read_data_attack(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_augmentation_data():
    all_data = []  # [img0, img1] <255
    all_label = []  # ['French bulldog', 'French bulldog', 'French bulldog']
    all_path = ['base_data.pkl', 'augmentation_flip_horizontal_vertical.pkl', 'augmentation_random_brightness.pkl',
                'augmentation_random_rotation.pkl', 'augmentation_random_translation.pkl', 'augmentation_random_zoom.pkl']
    all_path = [data_path+i for i in all_path]

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
    bim_data = read_data_attack(data_path+'bim_x_adv.pkl')
    fsgm_data = read_data_attack(data_path+'fsgm_x_adv.pkl')
    newf_data = read_data_attack(data_path+'newf_x_adv.pkl')
    patch_data = read_data_attack(data_path+'patch_x_adv.pkl')
    pgd_data = read_data_attack(data_path+'pgd_x_adv.pkl')
    sa_data = read_data_attack(data_path+'sa_x_adv.pkl')
    return bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data


def imitator_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(300, 300, 3)))
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
    model.add(Dense(127, name='dense_1'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


print('===============start to read data=====================')
all_data, all_label, attack_label = get_augmentation_data()
print('====================================')
bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data = get_attack_data()   # (9379, 300, 300, 3)
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


bim_train_pre = read_data(data_path+'bim_x_train_pre.pkl')
pgd_train_pre = read_data(data_path+'pgd_x_train_pre.pkl')
sa_train_pre = read_data(data_path+'sa_x_train_pre.pkl')
fsgm_train_pre = read_data(data_path+'fsgm_x_train_pre.pkl')
nf_train_pre = read_data(data_path+'newf_x_train_pre.pkl')
patch_train_pre = read_data(data_path+'patch_x_train_pre.pkl')

bim_test_pre = read_data(data_path+'bim_x_test_pre.pkl')
pgd_test_pre = read_data(data_path+'pgd_x_test_pre.pkl')
sa_test_pre = read_data(data_path+'sa_x_test_pre.pkl')
fsgm_test_pre = read_data(data_path+'fsgm_x_test_pre.pkl')
nf_test_pre = read_data(data_path+'newf_x_test_pre.pkl')
patch_test_pre = read_data(data_path+'patch_x_test_pre.pkl')


X_pre = np.concatenate((bim_train_pre, pgd_train_pre, sa_train_pre, fsgm_train_pre, nf_train_pre, patch_train_pre,
                        bim_test_pre, pgd_test_pre, sa_test_pre, fsgm_test_pre, nf_test_pre, patch_test_pre),
                        axis=0)
X_train = np.concatenate((bim_x_train, pgd_x_train, sa_x_train, fsgm_x_train, newf_x_train, patch_x_train,
                          bim_x_test, pgd_x_test, sa_x_test, fsgm_x_test, newf_x_test, patch_x_test),
                          axis=0)


model = imitator_model()


model.fit(X_train, X_pre, batch_size=64, shuffle=True, epochs=40, verbose=1)
model.save('imitator.h5')

