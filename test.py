import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle as pickle
import scipy.misc as misc


'''
def conv2d_padding_show():
    # [1, 13, 13, 2] ---> [m, height, width, channel]
    input = tf.Variable(tf.random_normal(X))
    # [6, 6, 2, 7] ---> [height, width, prev_channel, output_channel]
    filter = tf.Variable(tf.random_normal(conv2d_filter))

    op = tf.nn.conv2d(input, filter, strides=conv2d_strides, padding=pad)
    # VALID (1, 2, 2, 7)
    # SAME  (1, 3, 3, 7)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('image: ')
        image = sess.run(input)
        print(image.shape)

        print('pooling result: ')
        res = sess.run(op)
        print(res.shape)
pad = 'SAME'

# X ---> [m, height, width, channel]
# X = [1, 13, 13, 7]
X = [1, 7, 7, 151]

# ---> [1, f, f, 1]
# pooling_filter = [1, 6, 6, 1]
conv2d_filter = [4, 4, 512, 151]

# ---> [1, s, s, 1]
# conv2d_strides = [1, 5, 5, 1]
conv2d_strides = [1, 2, 2, 1]


# 自己改改 X, fileter, strides 的值，配合直觉经验，会有更好的理解

conv2d_padding_show()
'''
'''
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
data_dir = 'C:\\Users\ssp\Desktop\FCN.tensorflow-master\Data_zoo\MIT_SceneParsing'
SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])
pickle_filename = "MITSceneParsing.pickle"
pickle_filepath = os.path.join(data_dir, pickle_filename)
with open(pickle_filepath, 'rb') as f:
    result = pickle.load(f)
    training_records = result['training']
    validation_records = result['validation']
    print('ss')
'''
image_options = {'resize': True, 'resize_size': 224}
images = []
image1 = misc.imread('C:\\Users\ssp\Desktop\FCN.tensorflow-master\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\images\\training\ADE_train_00000010.jpg')#读为numpy
if len(image1.shape) < 3:
    image1 = np.array([image1 for i in range(3)])
if image_options.get("resize", False) and image_options["resize"]:
    resize_size = int(image_options["resize_size"])
    resize_image = misc.imresize(image1,
                                 [resize_size, resize_size], interp='nearest')
else:
    resize_image = image1
resize_image1 = np.array(resize_image)
images.append(resize_image1)

image2 = misc.imread('C:\\Users\ssp\Desktop\FCN.tensorflow-master\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\images\\training\ADE_train_00000011.jpg')#读为numpy
if len(image2.shape) < 3:
    image2 = np.array([image2 for i in range(3)])
if image_options.get("resize", False) and image_options["resize"]:
    resize_size = int(image_options["resize_size"])
    resize_image = misc.imresize(image2,
                                 [resize_size, resize_size], interp='nearest')
else:
    resize_image = image2
resize_image2 = np.array(resize_image)
images.append(resize_image2)
images = np.array(images)
print('ss')