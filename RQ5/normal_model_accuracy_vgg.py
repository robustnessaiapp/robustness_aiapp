import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model = load_model('vgg.h5')


def get_normal_score():
    normal_scores = model.evaluate(x_test, y_test, verbose=0)
    return normal_scores[1]


def get_attack_score(path_data):
    f = open(path_data, 'rb')
    data = pickle.load(f)
    scores = model.evaluate(data, y_test, verbose=0)
    return scores[1]


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def main():
    normal_score = get_normal_score()
    bim_score = get_attack_score('bim_x_test_adv.pkl')
    fsgm = get_attack_score('fsgm_x_test_adv.pkl')
    newf = get_attack_score('newf_x_test_adv.pkl')
    patch = get_attack_score('patch_x_test_adv.pkl')
    pgd = get_attack_score('pgd_x_test_adv.pkl')
    sa = get_attack_score('sa_x_test_adv.pkl')

    write_result(str(bim_score) + '>' + 'bim_score', 'normal_model_acc_vgg.txt')
    write_result(str(pgd) + '>' + 'pgd', 'normal_model_acc_vgg.txt')
    write_result(str(sa) + '>' + 'sa', 'normal_model_acc_vgg.txt')
    write_result(str(fsgm) + '>' + 'fsgm', 'normal_model_acc_vgg.txt')
    write_result(str(newf) + '>' + 'newf', 'normal_model_acc_vgg.txt')
    write_result(str(patch) + '>' + 'patch', 'normal_model_acc_vgg.txt')
    write_result(str(normal_score)+'>'+'normal_score', 'normal_model_acc_vgg.txt')


if __name__ == '__main__':
    main()


