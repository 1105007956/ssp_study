#coding=utf-8
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
from time import time
from sklearn.preprocessing import OneHotEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder("float", shape=[None, 10], name='label')


def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = p.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(10000, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)
        labels = np.array(labels)
        return images, labels


def load_cifar_data(data_dir):
    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir,'data_batch_%d' % (i+1))
        print('loading', f)
        image_batch, label_batch = load_cifar_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        xtrain = np.concatenate(images_train)
        ytrain = np.concatenate(labels_train)
        del image_batch, label_batch
    xtest, ytest = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    print("finished loading cifar-10 data")
    return xtrain, ytrain, xtest, ytest


def get_w(shape, regularizer):
    w = tf.Variable(tf.truncated_normal((shape), stddev=0.1, dtype=tf.float32))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_b(shape):
    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def get_train_batch(number, batch_size):
    return xtrain_normalize[number*batch_size:(number+1)*batch_size], ytrain_onehot[number*batch_size:(number+1)*batch_size]


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')


def forward(x,train,regularizer):
    w1 = get_w([5, 5, 3, 32], regularizer)
    b1 = get_b([32])
    conv1 = conv2d(x,w1)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
    pool1 = max_pool(relu1)

    w2 = get_w([5, 5, 32, 64], regularizer)
    b2 = get_b([64])
    conv2 = conv2d(pool1,w2)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    pool2 = max_pool(relu2)


    reshaped = tf.reshape(pool2, [-1, 4096])

    w3 = get_w([4096,128],regularizer)
    b3 = get_b([128])
    fc1 = tf.nn.relu(tf.matmul(reshaped, w3) + b3)
    if train: fc1 = tf.nn.dropout(fc1, 0.5)
    w4 = get_w([128, 512], regularizer)
    b4 = get_b([512])
    fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)
    w5 = get_w([512, 10], regularizer)
    b5 = get_b([10])
    y = tf.nn.softmax(tf.matmul(fc2, w5) + b5)
    return y


data_dir = 'D:/study/cifar-10/'
xtrain, ytrain, xtest, ytest = load_cifar_data(data_dir)
xtrain_normalize = xtrain.astype('float32') / 255
xtest_normalize = xtest.astype('float32') / 255
encoder = OneHotEncoder(sparse=False)
yy = [[0], [1], [2], [3], [4], [5], [6],  [7], [8], [9]]
encoder.fit(yy)
ytrain_reshape = ytrain.reshape(-1, 1)
ytrain_onehot = encoder.transform(ytrain_reshape)
ytest_reshape = ytest.reshape(-1, 1)
ytest_onehot = encoder.transform(ytest_reshape)

learning_rate_base = 0.0005
learning_rate_decay = 0.99
regularizer = 0.0001
steps = 800
moving_average_decay = 0.99
batch_size = 100
total_batch = int(len(xtrain) / batch_size)





y = forward(x, True, regularizer)
global_step = tf.Variable(0, trainable=False)
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cem = tf.reduce_mean(ce)
loss = cem + tf.add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, int(len(xtrain) / batch_size), learning_rate_decay, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
ema_op = ema.apply(tf.trainable_variables())

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.control_dependencies([optimizer, ema_op]):
    train_op = tf.no_op(name='train')
epoch = tf.Variable(0, name='epoch', trainable=False)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


ckpt_dir = 'D:/study/cifar-10'
saver = tf.train.Saver(max_to_keep=1)
ckpt = tf.train.latest_checkpoint(ckpt_dir)
if ckpt != None:
    saver.restore(sess, ckpt)
start = sess.run(epoch)
print('training from ',start, 'epoch')

for step in range(start, steps):
    for i in range(total_batch):
        x_batch, y_batch = get_train_batch(i, batch_size)
        _, loss_value, acc = sess.run([train_op, loss, accuracy], feed_dict={x:x_batch, y_:y_batch})
        if i % 100 ==0:
            print('loss:', loss_value, "accuracy:", acc)
    print('epoch:', step, 'loss:', loss_value, "accuracy:", acc)
    saver.save(sess,"D:/study/cifar-10/cifar10.ckpt")
    sess.run(epoch.assign(step+1))
print('train finished!')