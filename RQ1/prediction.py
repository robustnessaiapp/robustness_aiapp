import pickle
import numpy as np
import tensorflow as tf
import cv2


def read_data():
    f = open('dog.pkl', 'rb')
    res = pickle.load(f)
    data = res[0]
    label = res[1]
    return data, label


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


def main(path_embedded_model):
    data_list, label_list = read_data()
    model_path = path_embedded_model
    f = open("label.txt", "r")
    lines = f.readlines()
    label = [line.strip() for line in lines if len(line) > 1]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    pre_diction = []
    for img in data_list:
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        x = get_shape_1_300_300_3(img)

        output = interpreter.get_output_details()[0]
        input = interpreter.get_input_details()[0]
        interpreter.set_tensor(input['index'], x)
        interpreter.invoke()
        p_value = list(interpreter.get_tensor(output['index'])[0])
        dic = dict(zip(label, p_value))
        pre = sorted(dic.items(), key=lambda item: item[1], reverse=True)[0][0]
        pre_diction.append(pre)

    print(get_acc(label_list, pre_diction))


if __name__ == '__main__':
    main()