from metric_index import KMeans_index, KMeans_plus_plus_index, MiniBatchKMeans_index, AgglomerativeClustering_index, GaussianMixture_index, SpectralClustering_index
from metric_index import Margin_index, Variant_Margin_index, DeepGini_index, Variant_DeepGini_index, Entropy_index, Variant_Entropy_index, LeastConfidence_index, Variant_LeastConfidence_index, Variance_index, Variant_Variance_index
from metric_index import Dsa_index, Nc_index, BALD_index, EGL_index, Random_index
from tensorflow.keras.models import load_model
import pickle


file_name = 'index.txt'
model = load_model('imitator.h5')
layer_names = ['dense_1']



def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def read_data(path_pkl):
    f = open(path_pkl, 'rb')
    data = pickle.load(f)
    return data


x_select_pre = read_data('x_select_pre.pkl')
x_train = read_data('x_train.pkl')
x_select = read_data('x_select.pkl')
x = x_select_pre

try:
    Random_index_list = Random_index(x)
    write_result('Random_index_list->' + str(Random_index_list), file_name)
except:
    Random_index_list = 'wrong'
    write_result('Random_index_list->' + str(Random_index_list), file_name)

try:
    KMeans_index_list = KMeans_index(x)
    write_result('KMeans_index_list->' + str(KMeans_index_list), file_name)
except:
    KMeans_index_list = 'wrong'
    write_result('KMeans_index_list->' + str(KMeans_index_list), file_name)

try:
    KMeans_plus_plus_index_list = KMeans_plus_plus_index(x)
    write_result('KMeans_plus_plus_index_list->' + str(KMeans_plus_plus_index_list), file_name)
except:
    KMeans_plus_plus_index_list = 'wrong'
    write_result('KMeans_plus_plus_index_list->' + str(KMeans_plus_plus_index_list), file_name)

try:
    MiniBatchKMeans_index_list = MiniBatchKMeans_index(x)
    write_result('MiniBatchKMeans_index_list->' + str(MiniBatchKMeans_index_list), file_name)
except:
    MiniBatchKMeans_index_list = 'wrong'
    write_result('MiniBatchKMeans_index_list->' + str(MiniBatchKMeans_index_list), file_name)

try:
    AgglomerativeClustering_index_list = AgglomerativeClustering_index(x)
    write_result('AgglomerativeClustering_index_list->' + str(AgglomerativeClustering_index_list), file_name)
except:
    AgglomerativeClustering_index_list = 'wrong'
    write_result('AgglomerativeClustering_index_list->' + str(AgglomerativeClustering_index_list), file_name)

try:
    GaussianMixture_index_list = GaussianMixture_index(x)
    write_result('GaussianMixture_index_list->' + str(GaussianMixture_index_list), file_name)
except:
    GaussianMixture_index_list = 'wrong'
    write_result('GaussianMixture_index_list->' + str(GaussianMixture_index_list), file_name)

try:
    SpectralClustering_index_list = SpectralClustering_index(x)
    write_result('SpectralClustering_index_list->' + str(SpectralClustering_index_list), file_name)
except:
    SpectralClustering_index_list = 'wrong'
    write_result('SpectralClustering_index_list->' + str(SpectralClustering_index_list), file_name)

try:
    Margin_index_list = Margin_index(x)
    write_result('Margin_index_list->' + str(Margin_index_list), file_name)
except:
    Margin_index_list = 'wrong'
    write_result('Margin_index_list->' + str(Margin_index_list), file_name)

try:
    Variant_Margin_index_list = Variant_Margin_index(x)
    write_result('Variant_Margin_index_list->' + str(Variant_Margin_index_list), file_name)
except:
    Variant_Margin_index_list = 'wrong'
    write_result('Variant_Margin_index_list->' + str(Variant_Margin_index_list), file_name)

try:
    DeepGini_index_list = DeepGini_index(x)
    write_result('DeepGini_index_list->' + str(DeepGini_index_list), file_name)
except:
    DeepGini_index_list = 'wrong'
    write_result('DeepGini_index_list->' + str(DeepGini_index_list), file_name)

try:
    Variant_DeepGini_index_list = Variant_DeepGini_index(x)
    write_result('Variant_DeepGini_index_list->' + str(Variant_DeepGini_index_list), file_name)
except:
    Variant_DeepGini_index_list = 'wrong'
    write_result('Variant_DeepGini_index_list->' + str(Variant_DeepGini_index_list), file_name)

try:
    Entropy_index_list = Entropy_index(x)
    write_result('Entropy_index_list->' + str(Entropy_index_list), file_name)
except:
    Entropy_index_list = 'wrong'
    write_result('Entropy_index_list->' + str(Entropy_index_list), file_name)

try:
    Variant_Entropy_index_list = Variant_Entropy_index(x)
    write_result('Variant_Entropy_index_list->' + str(Variant_Entropy_index_list), file_name)
except:
    Variant_Entropy_index_list = 'wrong'
    write_result('Variant_Entropy_index_list->' + str(Variant_Entropy_index_list), file_name)

try:
    LeastConfidence_index_list = LeastConfidence_index(x)
    write_result('LeastConfidence_index_list->' + str(LeastConfidence_index_list), file_name)
except:
    LeastConfidence_index_list = 'wrong'
    write_result('LeastConfidence_index_list->' + str(LeastConfidence_index_list), file_name)

try:
    Variant_LeastConfidence_index_list = Variant_LeastConfidence_index(x)
    write_result('Variant_LeastConfidence_index_list->' + str(Variant_LeastConfidence_index_list), file_name)
except:
    Variant_LeastConfidence_index_list = 'wrong'
    write_result('Variant_LeastConfidence_index_list->' + str(Variant_LeastConfidence_index_list), file_name)

try:
    Variance_index_list = Variance_index(x)
    write_result('Variance_index_list->' + str(Variance_index_list), file_name)
except:
    Variance_index_list = 'wrong'
    write_result('Variance_index_list->' + str(Variance_index_list), file_name)

try:
    Variant_Variance_index_list = Variant_Variance_index(x)
    write_result('Variant_Variance_index_list->' + str(Variant_Variance_index_list), file_name)
except:
    Variant_Variance_index_list = 'wrong'
    write_result('Variant_Variance_index_list->' + str(Variant_Variance_index_list), file_name)


try:
    BALD_index_list = BALD_index(x_select)
    write_result('BALD_index_list->' + str(BALD_index_list), file_name)
except:
    BALD_index_list = 'wrong'
    write_result('BALD_index_list->' + str(BALD_index_list), file_name)


try:
    Dsa_index_list = Dsa_index(model, x_train, x_select, layer_names)
    write_result('Dsa_index_list->' + str(Dsa_index_list), file_name)
except:
    Dsa_index_list = 'wrong'
    write_result('Dsa_index_list->' + str(Dsa_index_list), file_name)

try:
    Nc_index_list = Nc_index(model, x_select)
    write_result('Nc_index_list->' + str(Nc_index_list), file_name)
except:
    Nc_index_list = 'wrong'
    write_result('Nc_index_list->' + str(Nc_index_list), file_name)


try:
    EGL_index_list = EGL_index(model, x_select)
    write_result('EGL_index_list->' + str(EGL_index_list), file_name)
except:
    EGL_index_list = 'wrong'
    write_result('EGL_index_list->' + str(EGL_index_list), file_name)




