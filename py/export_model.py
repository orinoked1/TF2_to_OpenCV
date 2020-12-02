
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os

# get a pre-trained ResNet50 model
base_model = tf.keras.applications.ResNet50(weights='imagenet')
model_input_shape = base_model.input_shape
# save model in SavedModel format
SavedModel_path = os.path.join('model', 'SavedModel_folder')
base_model.save(SavedModel_path, save_format='tf')
# load saved model and freeze it
loaded_model = tf.saved_model.load(SavedModel_path)
infer = loaded_model.signatures['serving_default']
concrete_function = tf.function(infer).get_concrete_function(input_1=tf.TensorSpec(shape=model_input_shape,
                                                                                   dtype=tf.float32))
wrapped_fun = convert_variables_to_constants_v2(concrete_function)
graph_def = wrapped_fun.graph.as_graph_def()
# save again in a *.pb file including weighs as constants
model_frozen_pb_full_path = os.path.join('model', 'frozenGraph_folder')
os.makedirs(model_frozen_pb_full_path, exist_ok=True)
with tf.io.gfile.GFile(os.path.join(model_frozen_pb_full_path,'frozenGraph.pb'), 'wb') as f:
    f.write(graph_def.SerializeToString())

import cv2 as cv
import numpy as np
# generate random input
image_shape = np.asarray(model_input_shape[1:])
input_shape = np.insert(image_shape, 0, 1, axis=0)  # add observation dim
test_input_c_last = np.random.standard_normal(input_shape).astype(np.float32)
test_input_c_first = np.moveaxis(test_input_c_last, -1, 1)
# inference in tensorflow
out_tf = base_model.predict(test_input_c_last)
# inference in openCV DNN
net = cv.dnn.readNet(os.path.join(model_frozen_pb_full_path,'frozenGraph.pb'))
net.setInput(test_input_c_first)
out_dnn = net.forward()
# compare feature vectors
np.testing.assert_allclose(out_tf, out_dnn, rtol=1e-03, atol=1e-05)

# save TF result to compare to Cpp
test_image_folder = 'img_folder'
os.makedirs(test_image_folder, exist_ok=True)
np.save(os.path.join(test_image_folder, 'TF_feature_vector.npy'), out_tf)
# export the test image to a Cpp project (using openCV *.xml file format)
# save the float array with shape [rows X cols X channels]
fs = cv.FileStorage(os.path.join(test_image_folder, 'test_img.xml'), cv.FILE_STORAGE_WRITE)
fs.write("test_img", np.squeeze(test_input_c_last))
fs.release()