from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


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


def get_shape_1_300_300_3(x_data):
    x = x_data
    x = x.reshape(1, 300, 300, 3)
    x = x.astype(np.float32)
    if np.max(x) > 5:
        x = x*1.0/255
    return x


def get_augmentation_data():
    all_data = []  # [img0, img1] <255
    all_label = []  # ['French bulldog', 'French bulldog', 'French bulldog']
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
    bim_data = read_data_attack('bim_x_adv.pkl')  # (9379, 300, 300, 3)
    fsgm_data = read_data_attack('fsgm_x_adv.pkl')
    newf_data = read_data_attack('newf_x_adv.pkl')
    patch_data = read_data_attack('patch_x_adv.pkl')
    pgd_data = read_data_attack('pgd_x_adv.pkl')
    sa_data = read_data_attack('sa_x_adv.pkl')
    return bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data



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
print('==============start tflite======================')

interpreter = tf.lite.Interpreter(model_path='dognet_a_dog_breed_identifier.tflite')
interpreter.allocate_tensors()


def get_prediction_pkl(X, path_pkl):
    pre_list = []
    for img in X:
        x = get_shape_1_300_300_3(img)
        output = interpreter.get_output_details()[0]
        input = interpreter.get_input_details()[0]
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        pre_list.append(p_value)
    pre_numpy = np.array(pre_list)
    output = open(path_pkl, 'wb')
    pickle.dump(pre_numpy, output)


get_prediction_pkl(all_data, 'all_data_pre.pkl')
print('===============1=====================')
get_prediction_pkl(bim_x_train, 'bim_x_train_pre.pkl')
print('===============2=====================')
get_prediction_pkl(fsgm_x_train, 'fsgm_x_train_pre.pkl')
print('===============3=====================')
get_prediction_pkl(newf_x_train, 'newf_x_train_pre.pkl')
print('===============4=====================')
get_prediction_pkl(patch_x_train, 'patch_x_train_pre.pkl')
print('===============5=====================')
get_prediction_pkl(pgd_x_train, 'pgd_x_train_pre.pkl')
print('===============6=====================')
get_prediction_pkl(sa_x_train, 'sa_x_train_pre.pkl')
print('===============7=====================')
get_prediction_pkl(bim_x_test, 'bim_x_test_pre.pkl')
print('===============8=====================')
get_prediction_pkl(fsgm_x_test, 'fsgm_x_test_pre.pkl')
print('===============9=====================')
get_prediction_pkl(newf_x_test, 'newf_x_test_pre.pkl')
print('===============10=====================')
get_prediction_pkl(patch_x_test, 'patch_x_test_pre.pkl')
print('===============11=====================')
get_prediction_pkl(pgd_x_test, 'pgd_x_test_pre.pkl')
print('===============12=====================')
get_prediction_pkl(sa_x_test, 'sa_x_test_pre.pkl')
print('=======finish======')



