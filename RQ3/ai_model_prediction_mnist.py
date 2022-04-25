import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import pickle

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)


model_path = 'hello_mnist_a_simple_digit_identifier.tflite'


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def get_shape_1_28_28_1(x):
    x = x.reshape(1, 28, 28, 1)
    x = x.astype(np.float32)
    return x


def model_prediction(x_test):
    predict_list = []
    for x in x_test:
        x = get_shape_1_28_28_1(x)
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        predict_list.append(p_value)
    return predict_list


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]

# data = read_data('sa_x_test_adv.pkl')
# model_prediction_list = model_prediction(data)   # [[], []]
# output_pkl = open('sa_x_test_adv_pre.pkl', 'wb')
# pickle.dump(model_prediction_list, output_pkl)

model_prediction_list = model_prediction(x_test)   # [[], []]
output_pkl = open('x_test_pre.pkl', 'wb')
pickle.dump(model_prediction_list, output_pkl)



