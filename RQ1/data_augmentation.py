import os
import pickle
import cv2
import numpy as np
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def read_data():
    f = open('dog.pkl', 'rb')
    res = pickle.load(f)
    data = res[0]
    label = res[1]
    return data, label


def get_shape_300_300_3(x_data):
    x = cv2.resize(x_data, (300, 300), cv2.INTER_AREA)
    x = x.reshape(300, 300, 3)
    x = x.astype(np.float32)
    return x


def get_data_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.jpg'):
                    path_list.append(file_absolute_path)
    return path_list


def augmentation_random_translation(img, label):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1)
    random_translational_data = []
    number = 15
    for i in range(number):
        batch = it.next()
        image = batch[0].astype('float')
        random_translational_data.append(image)  # 获取数据集
    label_list = [label] * number
    return random_translational_data, label_list


def main_augmentation_random_translation(data, label):
    re_data = []
    re_label = []
    for i in range(len(data)):
        data_list, label_list = augmentation_random_translation(data[i], label[i])
        re_data = re_data + data_list
        re_label = re_label + label_list

    res = [re_data, re_label]
    output = open('augmentation_random_translation.pkl', 'wb')
    pickle.dump(res, output)


def augmentation_flip_horizontal_vertical(img, label):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    it = datagen.flow(samples, batch_size=1)
    new_data = []
    number = 15
    for i in range(number):
        batch = it.next()
        image = batch[0].astype('float')
        new_data.append(image)
    label_list = [label] * number
    return new_data, label_list


def main_augmentation_flip_horizontal_vertical(data, label):
    re_data = []
    re_label = []
    for i in range(len(data)):
        data_list, label_list = augmentation_flip_horizontal_vertical(data[i], label[i])
        re_data = re_data + data_list
        re_label = re_label + label_list

    res = [re_data, re_label]
    output = open('augmentation_flip_horizontal_vertical.pkl', 'wb')
    pickle.dump(res, output)


def augmentation_random_rotation(img, label):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(rotation_range=45)
    it = datagen.flow(samples, batch_size=1)
    new_data = []
    number = 15
    for i in range(number):
        batch = it.next()
        image = batch[0].astype('float')
        new_data.append(image)
    label_list = [label] * number
    return new_data, label_list


def main_augmentation_random_rotation(data, label):
    re_data = []
    re_label = []
    for i in range(len(data)):
        data_list, label_list = augmentation_random_rotation(data[i], label[i])
        re_data = re_data + data_list
        re_label = re_label + label_list

    res = [re_data, re_label]
    output = open('augmentation_random_rotation.pkl', 'wb')
    pickle.dump(res, output)


def augmentation_random_brightness(img, label):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 3.0])
    it = datagen.flow(samples, batch_size=1)
    new_data = []
    number = 15
    for i in range(number):
        batch = it.next()
        image = batch[0].astype('int')
        new_data.append(image)
    label_list = [label] * number
    return new_data, label_list


def main_augmentation_random_brightness(data, label):
    re_data = []
    re_label = []
    for i in range(len(data)):
        data_list, label_list = augmentation_random_brightness(data[i], label[i])
        re_data = re_data + data_list
        re_label = re_label + label_list

    res = [re_data, re_label]
    output = open('augmentation_random_brightness.pkl', 'wb')
    pickle.dump(res, output)


def augmentation_random_zoom(img, label):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5, 1.0])
    it = datagen.flow(samples, batch_size=1)
    new_data = []
    number = 15
    for i in range(number):
        batch = it.next()
        image = batch[0].astype('float')
        new_data.append(image)
    label_list = [label] * number
    return new_data, label_list


def main_augmentation_random_zoom(data, label):
    re_data = []
    re_label = []
    for i in range(len(data)):
        data_list, label_list = augmentation_random_zoom(data[i], label[i])
        re_data = re_data + data_list
        re_label = re_label + label_list

    res = [re_data, re_label]
    output = open('augmentation_random_zoom.pkl', 'wb')
    pickle.dump(res, output)


if __name__ == '__main__':

    data, label = read_data()
    label_set = list(set(list(label)))
    select_label = ['Miniature schnauzer', 'French bulldog', 'Cocker spaniel', 'Irish wolfhound', 'Norwich terrier', 'Gordon setter', 'Bernese mountain dog', 'Great dane', 'Papillon', 'Norwegian elkhound']
    x_data = []
    y_label = []
    for i in range(len(label)):
        if label[i] in select_label:
            x_data.append(data[i])
            y_label.append(label[i])

    data = x_data
    label = y_label
    data = [get_shape_300_300_3(i) for i in data]

    main_augmentation_random_translation(data, label)
    main_augmentation_flip_horizontal_vertical(data, label)
    main_augmentation_random_rotation(data, label)
    main_augmentation_random_brightness(data, label)
    main_augmentation_random_zoom(data, label)



