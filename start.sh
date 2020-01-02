python3 vgg16.py
rm -rf model
mkdir model
tensorflowjs_converter --input_format keras vgg16.h5 model/

