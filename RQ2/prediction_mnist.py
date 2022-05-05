import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import pickle
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_shape_1_28_28_1(x):
    x = x.reshape(1, 28, 28, 1)
    x = x.astype(np.float32)
    return x


def get_shape_1_10(x):
    x = x.reshape(1, 10)
    x = x.astype(np.float32)
    return x


def model_prediction(x_mobile, x_data):
    predict_list = []
    for i in range(len(x_data)):
        x1 = x_mobile[i]
        x1 = get_shape_1_10(x1)
        x2 = x_data[i]
        x2 = get_shape_1_28_28_1(x2)
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
    interpreter = tf.lite.Interpreter(model_path='patch_model.tflite')
    interpreter.allocate_tensors()
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()


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

    x_test_pre = read_data('x_test_pre.pkl')
    x_test_pre = np.array(x_test_pre)

    data_pre_list = [bim_x_test_adv_pre, fsgm_x_test_adv_pre, newf_x_test_adv_pre, patch_x_test_adv_pre, pgd_x_test_adv_pre, sa_x_test_adv_pre, x_test_pre]
    data_list = [bim_x_test_adv, fsgm_x_test_adv, newf_x_test_adv, patch_x_test_adv, pgd_x_test_adv, sa_x_test_adv, x_test]
    name_list = ['bim_x_test_adv', 'fsgm_x_test_adv', 'newf_x_test_adv', 'patch_x_test_adv', 'pgd_x_test_adv', 'sa_x_test_adv', 'x_test']
    for i in range(len(data_list)):
        model_prediction_list = model_prediction(data_pre_list[i], data_list[i])
        pre_list = np.argmax(model_prediction_list, axis=1)
        label_list = np.argmax(y_test, axis=1)
        acc = accuracy_score(pre_list, label_list)
        attack_name = name_list[i]
        print(acc, attack_name)
        write_result(str(acc)+'->'+attack_name, 'patch_model_prediction.txt')

