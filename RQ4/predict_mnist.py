import numpy as np
import tensorflow as tf
import os
import pickle
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
from tensorflow.keras.datasets import mnist


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_shape_1_10(x):
    x = x.reshape(1, 10)
    x = x.astype(np.float32)
    return x


def get_shape_1_28_28_1(x):
    x = x.reshape(1, 28, 28, 1)
    x = x.astype(np.float32)
    return x


def get_ai_model(path_dir):
    model_path_list = []
    if os.path.isdir(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=True):
            for file in files:
                if '_index_list_' in file and file.endswith('.tflite'):
                    model_path_list.append(file)
    return model_path_list


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test, 10)
y_train = tf.keras.utils.to_categorical(y_train, 10)


bim_x_test_adv_pre = read_data('bim_x_test_adv_pre.pkl')
bim_x_test_adv_pre = np.array(bim_x_test_adv_pre)
bim_x_test_adv = read_data('bim_x_test_adv.pkl')

fsgm_x_test_adv_pre = read_data('fsgm_x_test_adv_pre.pkl')
fsgm_x_test_adv_pre = np.array(fsgm_x_test_adv_pre)
fsgm_x_test_adv = read_data('fsgm_x_test_adv.pkl')

newf_x_test_adv_pre = read_data('newf_x_test_adv_pre.pkl')
newf_x_test_adv_pre = np.array(newf_x_test_adv_pre)
newf_x_test_adv = read_data('newf_x_test_adv.pkl')

patch_x_test_adv_pre = read_data('patch_x_test_adv_pre.pkl')
patch_x_test_adv_pre = np.array(patch_x_test_adv_pre)
patch_x_test_adv = read_data('patch_x_test_adv.pkl')

pgd_x_test_adv_pre = read_data('pgd_x_test_adv_pre.pkl')
pgd_x_test_adv_pre = np.array(pgd_x_test_adv_pre)
pgd_x_test_adv = read_data('pgd_x_test_adv.pkl')

sa_x_test_adv_pre = read_data('sa_x_test_adv_pre.pkl')
sa_x_test_adv_pre = np.array(sa_x_test_adv_pre)
sa_x_test_adv = read_data('sa_x_test_adv.pkl')


all_test = np.concatenate((bim_x_test_adv, fsgm_x_test_adv, newf_x_test_adv, patch_x_test_adv, pgd_x_test_adv, sa_x_test_adv), axis=0)
all_test_pre = np.concatenate((bim_x_test_adv_pre, fsgm_x_test_adv_pre, newf_x_test_adv_pre, patch_x_test_adv_pre, pgd_x_test_adv_pre, sa_x_test_adv_pre), axis=0)
all_y_test = np.concatenate((y_test, y_test, y_test, y_test, y_test, y_test), axis=0)


def main_single(model_path_list):
    for model_path in model_path_list:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        output = interpreter.get_output_details()[0]
        input = interpreter.get_input_details()

        predict_list = []
        for i in range(len(all_test)):
            x1 = all_test_pre[i]
            x1 = get_shape_1_10(x1)
            x2 = all_test[i]
            x2 = get_shape_1_28_28_1(x2)
            interpreter.set_tensor(input[0]['index'], x1)
            interpreter.set_tensor(input[1]['index'], x2)
            interpreter.invoke()
            p_value = list(interpreter.get_tensor(output['index'])[0])
            predict_list.append(p_value)

        pre_list = [np.argmax(i) for i in predict_list]
        real_list = [np.argmax(i) for i in all_y_test]
        acc = accuracy_score(real_list, pre_list)
        content = model_path+'='+str(acc)
        write_result(str(content), 'predict.log')


def main(model_path_list, number_single, number_pool):
    number_group = len(model_path_list) // number_single + 1
    start = 0
    end = number_group
    data_list = []
    if start < len(model_path_list):
        for i in range(number_group):
            tmp = model_path_list[start:end]
            data_list.append(tmp)
            start = end
            end = end+number_single

    with Pool(number_pool) as p:
        p.map(main_single, data_list)


if __name__ == '__main__':

    path_dir = 'mnist_digit_identifier'
    model_path_list = get_ai_model(path_dir)
    main(model_path_list, 13, 28)



