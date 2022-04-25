from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import load_model
import numpy as np
import pickle


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


def read_index(path_index):
    f = open(path_index, "r")
    lines = f.readlines()
    lines = [i for i in lines if len(i) > 10 and 'wrong' not in i]

    name_metric_list = [i.split('->')[0].strip() for i in lines]
    index_list = [i.split('->')[-1].strip() for i in lines]
    index_list = [eval(i) for i in index_list]
    dic = dict(zip(name_metric_list, index_list))
    return dic


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


dic = read_index('index.txt')
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


def get_samples_list(x_select):
    ratio = [0.10, 0.20, 0.30, 0.40]
    number_samples = [int(len(x_select)*i) for i in ratio]
    index_start_end = [[0, number_samples[0]]]
    for i in range(1, len(number_samples)):
        start = number_samples[i-1]
        end = number_samples[i]
        tmp = [start, end]
        index_start_end.append(tmp)
    return index_start_end


def main(path_index, path_model, name_metric):
    dic = read_index(path_index)
    # x_train, x_select, x_test, y_train, y_select, y_test = get_data()
    log = 1
    model = load_model(path_model)
    select_times = 1
    index_start_end = get_samples_list(dic[name_metric])
    for start_end in index_start_end:
        id_list = dic[name_metric][0:start_end[1]]
        x = x_select[id_list]
        y = y_select[id_list]
        x_pre = x_select_pre[id_list]

        x = np.concatenate((x, x_train), axis=0)
        y = np.concatenate((y, y_train), axis=0)

        x_pre = np.concatenate((x_pre, x_train_pre), axis=0)

        model.fit([x_pre, x], y, validation_data=([x_test_pre, x_test], y_test),
                  shuffle=True, epochs=5, batch_size=64)

        save_model_name = name_metric+'_'+str(select_times)+'.h5'
        model.save(save_model_name)
        log += 1
        write_result(str(log), 'retrain.log')
        select_times += 1


metric_list = ['Random_index_list', 'KMeans_index_list', 'Variant_Margin_index_list', 'MiniBatchKMeans_index_list',
               'Margin_index_list', 'KMeans_plus_plus_index_list', 'DeepGini_index_list', 'Variant_Entropy_index_list',
               'Entropy_index_list', 'Variant_DeepGini_index_list', 'LeastConfidence_index_list', 'GaussianMixture_index_list',
               'Variance_index_list', 'Variant_LeastConfidence_index_list', 'Variant_Variance_index_list', 'BALD_index_list',
               'Nc_index_list']

path_index = 'index.txt'
path_model = 'patch_model.h5'
name_metric = metric_list[0]
main(path_index, path_model, name_metric)


