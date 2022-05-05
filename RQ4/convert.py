import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_file='alexnet.h5')
converter.post_training_quantize = True


tflite_model = converter.convert()
open('alexnet.tflite', "wb").write(tflite_model)
