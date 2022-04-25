import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score
import os


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_shape_1_300_300_3(x_data):
    x = x_data
    x = x.reshape(1, 300, 300, 3)
    x = x.astype(np.float32)
    x = x*1.0/255
    return x


def get_acc(label, real):
    n = 0
    for i in range(len(label)):
        if label[i] == real[i]:
            n += 1
    return n*1.0/len(label)


def get_ai_model(path_dir):
    model_path_list = []
    if os.path.isdir(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=True):
            for file in files:
                if '_index_list_' in file and file.endswith('.tflite'):
                    model_path_list.append(file)
    return model_path_list


def get_shape_1_127(x):
    x = x.reshape(1, 127)
    x = x.astype(np.float32)
    return x


x_select = read_data('x_select.pkl')
x_select_pre = read_data('x_select_pre.pkl')
x_test = read_data('X_test.pkl')
x_test_pre = read_data('X_test_pre.pkl')
x_train = read_data('x_train.pkl')
x_train_pre = read_data('x_train_pre.pkl')
y_select = read_data('y_select.pkl')
y_select_pre = read_data('y_select_pre.pkl')
y_test = read_data('Y_test.pkl')
y_train = read_data('y_train.pkl')


interpreter = tf.lite.Interpreter(model_path='patch_model.tflite')
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()

predict_list = []
for i in range(len(x_test)):
    x1 = x_test_pre[i]
    x1 = get_shape_1_127(x1)
    x2 = x_test[i]
    x2 = get_shape_1_300_300_3(x2)
    interpreter.set_tensor(input[0]['index'], x1)
    interpreter.set_tensor(input[1]['index'], x2)
    interpreter.invoke()
    p_value = list(interpreter.get_tensor(output['index'])[0])
    predict_list.append(p_value)

pre_list = [np.argmax(i) for i in predict_list]
real_list = [np.argmax(i) for i in y_test]
acc = accuracy_score(real_list, pre_list)
content = 'patch_model.tflite' + '=' + str(acc)
write_result(str(content), 'test_predict.log')









