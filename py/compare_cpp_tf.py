import cv2 as cv
import numpy as np
import os

test_image_folder = 'img_folder'

tf_out = np.load(os.path.join(test_image_folder, 'TF_feature_vector.npy'))
openCV_file = cv.FileStorage(os.path.join(test_image_folder,'cpp_feature_vector.xml'), cv.FILE_STORAGE_READ)
openCV_out = openCV_file.getFirstTopLevelNode().mat()
np.testing.assert_allclose(tf_out, openCV_out, rtol=1e-03, atol=1e-05)
max_abs_diff = np.max(np.abs(tf_out-openCV_out))
max_rel_diff = np.max(np.abs((tf_out-openCV_out)/tf_out))
print('max absolute difference is {:e} max relative difference is {:e}'.format(max_abs_diff,max_rel_diff))