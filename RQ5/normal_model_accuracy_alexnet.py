import pickle
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)
model = load_model('alexnet.h5')


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

    write_result(str(bim_score) + '>' + 'bim_score', 'normal_model_acc_alexnet.txt')
    write_result(str(pgd) + '>' + 'pgd', 'normal_model_acc_alexnet.txt')
    write_result(str(sa) + '>' + 'sa', 'normal_model_acc_alexnet.txt')
    write_result(str(fsgm) + '>' + 'fsgm', 'normal_model_acc_alexnet.txt')
    write_result(str(newf) + '>' + 'newf', 'normal_model_acc_alexnet.txt')
    write_result(str(patch) + '>' + 'patch', 'normal_model_acc_alexnet.txt')
    write_result(str(normal_score)+'>'+'normal_score', 'normal_model_acc_alexnet.txt')


if __name__ == '__main__':
    main()


