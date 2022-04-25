from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
    bim_data = read_data_attack('bim_x_adv.pkl')  
    fsgm_data = read_data_attack('fsgm_x_adv.pkl')
    newf_data = read_data_attack('newf_x_adv.pkl')
    patch_data = read_data_attack('patch_x_adv.pkl')
    pgd_data = read_data_attack('pgd_x_adv.pkl')
    sa_data = read_data_attack('sa_x_adv.pkl')
    return bim_data, fsgm_data, newf_data, patch_data, pgd_data, sa_data


def get_shape_1_127(x):
    x = x.reshape(1, 127)
    x = x.astype(np.float32)
    return x


def get_shape_1_300_300_3(x_data):
    x = x_data
    x = x.reshape(1, 300, 300, 3)
    x = x.astype(np.float32)
    if np.max(x) > 5:
        x = x*1.0/255
    return x


def model_prediction(x_mobile, x_data):
    predict_list = []
    for i in range(len(x_data)):
        x1 = x_mobile[i]
        x1 = get_shape_1_127(x1)
        x2 = x_data[i]
        x2 = get_shape_1_300_300_3(x2)
        interpreter.set_tensor(input[0]['index'], x1)
        interpreter.set_tensor(input[1]['index'], x2)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        predict_list.append(p_value)
    return predict_list


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


if __name__ == '__main__':
    print('===============start to read data=====================')
    all_data, all_label, attack_label = get_augmentation_data()
    print('====================================')
    all_x_train, all_x_test, all_y_train, all_y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=5)
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

    bim_x_test_pre = read_data_pre('bim_x_test_pre.pkl')
    fsgm_x_test_pre = read_data_pre('fsgm_x_test_pre.pkl')
    newf_x_test_pre = read_data_pre('newf_x_test_pre.pkl')
    patch_x_test_pre = read_data_pre('patch_x_test_pre.pkl')
    pgd_x_test_pre = read_data_pre('pgd_x_test_pre.pkl')
    sa_x_test_pre = read_data_pre('sa_x_test_pre.pkl')

    interpreter = tf.lite.Interpreter(model_path='patch_model.tflite')
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()

    model_prediction_list = model_prediction(bim_x_test_pre, bim_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(bim_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'BIM')
    write_result(str(acc)+'->'+'BIM', 'patch_model_prediction.txt')

    model_prediction_list = model_prediction(fsgm_x_test_pre, fsgm_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(fsgm_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'FSGM')
    write_result(str(acc)+'->'+'FSGM', 'patch_model_prediction.txt')

    model_prediction_list = model_prediction(newf_x_test_pre, newf_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(newf_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'NEWF')
    write_result(str(acc)+'->'+'NEWF', 'patch_model_prediction.txt')

    model_prediction_list = model_prediction(patch_x_test_pre, patch_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(patch_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'patch')
    write_result(str(acc)+'->'+'patch', 'patch_model_prediction.txt')

    model_prediction_list = model_prediction(pgd_x_test_pre, pgd_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(pgd_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'pgd')
    write_result(str(acc)+'->'+'pgd', 'patch_model_prediction.txt')

    model_prediction_list = model_prediction(sa_x_test_pre, sa_x_test)
    pre_list = np.argmax(model_prediction_list, axis=1)
    label_list = np.argmax(sa_y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    print(acc, 'sa')
    write_result(str(acc)+'->'+'sa', 'patch_model_prediction.txt')
