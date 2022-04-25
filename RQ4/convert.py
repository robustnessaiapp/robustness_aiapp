import tensorflow as tf
import os
from multiprocessing import Pool


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def convertion(path_h5):
    mobile_model_name = path_h5.split('.')[0].strip() + '.tflite'
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file=path_h5)
    converter.post_training_quantize = True

    tflite_model = converter.convert()
    open(mobile_model_name, "wb").write(tflite_model)


def get_ai_model(path_dir):
    model_path_list = []
    if os.path.isdir(path_dir):
        for root, dirs, files in os.walk(path_dir, topdown=True):
            for file in files:
                if '_index_list_' in file and file.endswith('.h5'):
                    model_path_list.append(file)
    return model_path_list


def main_single(model_path_list):
    log = 1
    for path_h5 in model_path_list:
        convertion(path_h5)
        write_result(str(log), 'convert.log')
        log += 1


def main(metric_list, number_single, number_pool):
    number_group = len(metric_list) // number_single + 1

    start = 0
    end = number_group
    data_list = []
    if start < len(metric_list):
        for i in range(number_group):
            tmp = metric_list[start:end]
            data_list.append(tmp)
            start = end
            end = end+number_single

    with Pool(number_pool) as p:
        p.map(main_single, data_list)


if __name__ == '__main__':
    path = 'dog_breed_identifier'
    metric_list = get_ai_model(path)
    main(metric_list, 13, 28)



