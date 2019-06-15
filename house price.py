#coding=utf-8
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
df = pd.read_csv("D:/study/boston.csv", header=0)
#print(df.describe())
df = df.values
df = np.array(df)

for i in range(12):
    df[:, i] = df[:, i]/(df[:, i].max() - df[:, i].min())

x_data = df[:, :12]
y_data = df[:, 12]

x = tf.placeholder(tf.float32, [None, 12], name="x")
y = tf.placeholder(tf.float32, [None, 1],  name="y")
with tf.name_scope("fc"):
    w = tf.Variable(tf.truncated_normal([12, 16], stddev=0.01), name="w")
    b = tf.Variable(tf.constant(0.1, tf.float32,[16]), name="b")



pred = tf.matmul(x, w) + b
train_epochs = 1000
learning_rate = 0.01
loss_list = []
with tf.name_scope("Loss_Function"):
    loss_function = tf.reduce_mean(tf.pow(y-pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1, 1)
        _, loss = sess.run([optimizer, loss_function], feed_dict={x: xs, y: ys})
        loss_sum = loss_sum + loss
    x_data, y_data = shuffle(x_data, y_data)
    loss_average = loss_sum/len(y_data)
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_list.append(loss_average)
    print("epoch=", epoch+1, "loss=", loss_average)

plt.plot(loss_list)
n = np.random.randint(506)
x_test = x_data[n]
x_test = x_test.reshape(1, 12)
predict = sess.run(pred, feed_dict={x: x_test})
target = y_data[n]
print("预测值：", predict, "标签值：", target)
