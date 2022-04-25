import math
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, Birch, SpectralClustering, FeatureAgglomeration, MeanShift
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy import stats
import sa as sa
from ncoverage import NCoverage


def get_cluster_index(x, label_list):
    df = pd.DataFrame(x)
    df['label'] = label_list
    center_list = []
    for i in list(set(label_list)):
        tmp_df = df[df['label'] == i].copy()
        del tmp_df["label"]
        tmp_x = tmp_df.to_numpy()
        center = np.average(tmp_x, axis=0)
        center_list.append(center)

    index_distance = []
    for i in range(len(x)):
        tmp = 0
        for p in center_list:
            distances = np.linalg.norm(x[i] - p)
            tmp = min(tmp, distances)
        index_distance.append(tmp)
    index_distance = np.array(index_distance)
    index_list = list(np.argsort(index_distance))
    return index_list


def Random_index(x):
    index_index = random.sample(list(range(len(x))), len(x))
    return index_index


def KMeans_index(x):
    kmeans = KMeans(n_clusters=10, init='random', random_state=0).fit(x)
    label_list = list(kmeans.labels_)
    index_list = get_cluster_index(x, label_list)
    return index_list


def KMeans_plus_plus_index(x):
    kmeans = KMeans(n_clusters=10, init='k-means++', random_state=1).fit(x)
    label_list = list(kmeans.labels_)
    index_list = get_cluster_index(x, label_list)
    return index_list


def MiniBatchKMeans_index(x):
    MBK = MiniBatchKMeans(n_clusters=10, random_state=2).fit(x)
    label_list = list(MBK.labels_)
    index_list = get_cluster_index(x, label_list)
    return index_list


def AgglomerativeClustering_index(x):
    AC = AgglomerativeClustering(n_clusters=10)
    AC.fit(x)
    label_list = list(AC.labels_)
    index_list = get_cluster_index(x, label_list)
    return index_list


def GaussianMixture_index(x):
    GM = GaussianMixture(n_components=10, random_state=3)
    GM.fit(x)
    label_list = list(GM.predict(x))
    index_list = get_cluster_index(x, label_list)
    return index_list


def SpectralClustering_index(x):
    SP = SpectralClustering(n_clusters=10, random_state=4)
    SP.fit(x)
    label_list = list(SP.labels_)
    index_list = get_cluster_index(x, label_list)
    return index_list


###############################################
def Margin_index(x):
    output_sort = np.sort(x)
    margin_score = output_sort[:, -1] - output_sort[:, -2]
    select_index = np.argsort(margin_score)
    return list(select_index)


def Variant_Margin_index(x):
    output_sort = np.sort(x)
    variant_margin_score = output_sort[:, -1] - output_sort[:, -2] + output_sort[:, -1] - output_sort[:, -3]
    select_index = np.argsort(variant_margin_score)
    return list(select_index)


def DeepGini_index(x):
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    select_index = np.argsort(gini_score)[::-1]
    return list(select_index)


def Variant_DeepGini_index(x):
    new_x = np.sort(x)[:, -3:]
    gini_score = 1 - np.sum(np.power(new_x, 2), axis=1)
    select_index = np.argsort(gini_score)[::-1]
    return list(select_index)


def Entropy_index(x):
    score = -np.sum(x * np.log(x), axis=1)
    select_index = np.argsort(score)[::-1]
    return list(select_index)


def Variant_Entropy_index(x):
    new_x = np.sort(x)[:, -3:]
    score = -np.sum(new_x * np.log(new_x), axis=1)
    select_index = np.argsort(score)[::-1]
    return list(select_index)


def LeastConfidence_index(x):
    max_pre = x.max(1)
    select_index = np.argsort(max_pre)
    return list(select_index)


def Variant_LeastConfidence_index(x):
    min_pre = x.min(1)
    select_index = np.argsort(min_pre)[::-1]
    return list(select_index)


def Variance_index(x):
    var = np.var(x, axis=1)
    select_index = np.argsort(var)
    return list(select_index)


def Variant_Variance_index(x):
    new_x = np.sort(x)[:, -3:]
    var = np.var(new_x, axis=1)
    select_index = np.argsort(var)
    return list(select_index)


# layer_names: ['dense_1']
def Dsa_index(model, x_train, x_select, layer_names):
    dsascores = sa.fetch_dsa(model, x_train, x_select, "candidates", layer_names, num_classes=127,
                             var_threshold=1e-5,
                             is_classification=True)
    return list(np.argsort(dsascores)[::-1])


def Nc_index(model, x_select):
    ncComputor = NCoverage(model, threshold=0.2)
    nc_score = np.array([])

    for i in range(0, x_select.shape[0], 100):
        if i + 100 > x_select.shape[0]:
            new_score = ncComputor.batch_nc(x_select[i:])
        else:
            new_score = ncComputor.batch_nc(x_select[i: i + 100])
        nc_score = np.append(nc_score, new_score)
    return list(np.argsort(nc_score)[::-1])


def BALD_index(x):
    BALD_list = []
    mode_list = []
    data_len = len(x)
    for _ in range(20):
        prediction = np.argmax(x, axis=1)
        BALD_list.append(prediction)

    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1,))[1][0]
        mode_list.append(1 - mode_num / 50)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[::-1]
    return list(select_index)


import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy


def EGL_compute(model, unlabeled, n_classes):
    model(tf.keras.Input((model.input_shape[-3], model.input_shape[-2], model.input_shape[-1])))
    input_placeholder = K.placeholder(model.get_input_shape_at(0))
    output_placeholder = K.placeholder(model.get_output_shape_at(0))
    predict = model.call(input_placeholder)
    loss = K.mean(categorical_crossentropy(output_placeholder, predict))
    weights = [tensor for tensor in model.trainable_weights]
    gradient = model.optimizer.get_gradients(loss, weights)
    gradient_flat = [K.flatten(x) for x in gradient]
    gradient_flat = K.concatenate(gradient_flat)
    gradient_length = tf.keras.backend.sum(K.square(gradient_flat))
    get_gradient_length = K.function([input_placeholder, output_placeholder], [gradient_length])
    unlabeled_predictions = model.predict(unlabeled)

    egls = np.zeros(unlabeled.shape[0])
    for i in range(n_classes):
        calculated_so_far = 0
        while calculated_so_far < unlabeled_predictions.shape[0]:
            if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                next = unlabeled_predictions.shape[0] - calculated_so_far
            else:
                next = 100

            labels = np.zeros((next, n_classes))
            labels[:, i] = 1
            # with eager_learning_phase_scope(value=0):
            grads = get_gradient_length([unlabeled[calculated_so_far:calculated_so_far + next, :], labels])[0]
            grads *= unlabeled_predictions[calculated_so_far:calculated_so_far + next, i]
            egls[calculated_so_far:calculated_so_far + next] += grads

            calculated_so_far += next

    return egls


def EGL_index(model, target_data):
    n_classes = 127
    egls = EGL_compute(model, target_data, n_classes)
    select_index = np.argsort(egls)[::-1]
    return list(select_index)
