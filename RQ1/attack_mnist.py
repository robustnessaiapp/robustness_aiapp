import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import pickle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import *


batch_size = 64
nb_epochs = 10
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = to_categorical(y_test, 10)
y_train = to_categorical(y_train, 10)


def Lenet1():
    model = Sequential()
    model.add(Conv2D(4, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), name='block1_pool1'))
    model.add(Conv2D(12, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal', name='dense_1'))
    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def get_train_model(model, x_train, y_train, batch_size, nb_epochs, save_model_path):
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=True)
    model.save(save_model_path)
    return classifier

model = Lenet1()
classifier = get_train_model(model, x_train, y_train, batch_size, nb_epochs, 'lenet1.h5')


def fgm_x_adv(classifier):
    attack = FastGradientMethod(estimator=classifier, eps=0.1)  
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('fsgm_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('fsgm_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def patch_x_adv(classifier):
    attack = AdversarialPatch(classifier=classifier)
    x_test_adv = attack.apply_patch(x=x_test, scale=0.3)
    x_train_adv = attack.apply_patch(x=x_train, scale=0.3)
    output = open('patch_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('patch_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def bim_x_adv(classifier):
    attack = BasicIterativeMethod(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('bim_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('bim_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def pgd_x_adv(classifier):
    attack = ProjectedGradientDescent(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('pgd_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('pgd_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def newf_x_adv(classifier):
    attack = NewtonFool(classifier=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('newf_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('newf_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


def sa_x_adv(classifier):
    attack = SquareAttack(estimator=classifier)
    x_test_adv = attack.generate(x=x_test)
    x_train_adv = attack.generate(x=x_train)
    output = open('sa_x_test_adv.pkl', 'wb')
    pickle.dump(x_test_adv, output)
    output = open('sa_x_train_adv.pkl', 'wb')
    pickle.dump(x_train_adv, output)


if __name__ == '__main__':
    fgm_x_adv(classifier)
    patch_x_adv(classifier)
    bim_x_adv(classifier)
    pgd_x_adv(classifier)
    newf_x_adv(classifier)
    sa_x_adv(classifier)
