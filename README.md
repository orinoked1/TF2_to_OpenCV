# TF2_to_OpenCV
This is an example code for exporting models from TS 2.x to OpenCV DNN (in python\C++)

# python requirements 
install a python 3.7 environment with TensorFlow 2.1 and openCV 4.4
(a fitting conda environment can be crated using TF2_to_OpenCV\py\environment.yml)

# C++ requirments 
A C++ project with OpenCV 4.4 
(if you choose to use the project in this repository make sure to define an environment variable named OPENCV_DIR to openCV bin directory
https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html)

# run 
first, run the export_model.py script it will create the model files as well as the random image for testing 
than tun the C++ code to generate the c++ feature vector file 
finally, the compare_cpp_tf.py to compare between cpp to TF
