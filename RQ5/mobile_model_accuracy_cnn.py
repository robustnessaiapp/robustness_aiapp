import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from sklearn.metrics import accuracy_score

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


model_path = 'cnn.tflite'


def get_shape_1_32_32_3(x):
    x = x.reshape(1, 32, 32, 3)
    x = x.astype(np.float32)
    return x


def mobile_model_prediction(data_test):
    predict_list = []
    for x in data_test:
        x = get_shape_1_32_32_3(x)
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        predict_list.append(p_value)
    pre_list = np.argmax(predict_list, axis=1)
    label_list = np.argmax(y_test, axis=1)
    acc = accuracy_score(pre_list, label_list)
    return acc


def mobile_attack_prediction(path_data):
    f = open(path_data, 'rb')
    data_test = pickle.load(f)
    acc = mobile_model_prediction(data_test)
    return acc


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]


def main():
    mobile_score = mobile_model_prediction(x_test)
    bim_score = mobile_attack_prediction('bim_x_test_adv.pkl')
    fsgm = mobile_attack_prediction('fsgm_x_test_adv.pkl')
    newf = mobile_attack_prediction('newf_x_test_adv.pkl')
    patch = mobile_attack_prediction('patch_x_test_adv.pkl')
    pgd = mobile_attack_prediction('pgd_x_test_adv.pkl')
    sa = mobile_attack_prediction('sa_x_test_adv.pkl')

    write_result(str(bim_score) + '>' + 'bim_score', 'mobile_model_acc_cnn.txt')
    write_result(str(pgd) + '>' + 'pgd', 'mobile_model_acc_cnn.txt')
    write_result(str(sa) + '>' + 'sa', 'mobile_model_acc_cnn.txt')
    write_result(str(fsgm) + '>' + 'fsgm', 'mobile_model_acc_cnn.txt')
    write_result(str(newf) + '>' + 'newf', 'mobile_model_acc_cnn.txt')
    write_result(str(patch) + '>' + 'patch', 'mobile_model_acc_cnn.txt')
    write_result(str(mobile_score)+'>'+'mobile_score', 'mobile_model_acc_cnn.txt')


if __name__ == '__main__':
    main()


