import tensorflow as tf
import pandas as pd
import os #changed
from matplotlib import pyplot as plt
import input_data

os.chdir(r'C:\Users\drose\Documents\GitHub\hello-tensorflow') #changed
#read data from csv
train_data = pd.read_csv("iris_training.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'], skiprows=1) #changed

test_data = pd.read_csv("iris_test.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'], skiprows=1) #changed
train_data.head() #changed

    #encode results to onehot
encodings = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]} #changed
train_data['f5'] = train_data['f5'].map(encodings) #changed
test_data['f5'] = test_data['f5'].map(encodings) #changed
train_data.head() #changed

#separate train data
train_x = train_data[['f1', 'f2', 'f3', 'f4']]
train_y = train_data.loc[:, 'f5'] #changed

#separate test data
test_x = test_data[['f1', 'f2', 'f3', 'f4']]
test_y = test_data.loc[:, 'f5']

#placeholders for inputs and outputs
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

#weight and bias
weight = tf.Variable(tf.zeros([4, 3]))
bias = tf.Variable(tf.zeros([3]))

#output after going activation function
output = tf.nn.softmax(tf.matmul(X, weight) + bias)
#cost funciton
cost = tf.reduce_mean(tf.square(Y-output))
#train model
train = tf.train.AdamOptimizer(0.01).minimize(cost)

#check sucess and failures
success = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
#calculate accuracy
accuracy = tf.reduce_mean(tf.cast(success, tf.float32))*100

#initialize variables
init = tf.global_variables_initializer()

#start the tensorflow session
with tf.Session() as sess:
    costs = []
    sess.run(init)
    #train model 1000 times
    for i in range(1000):
        _,c = sess.run([train, cost], {X: train_x, Y: [t for t in train_y.as_matrix()]})
        costs.append(c)

    print("Training finished!")

    #plot cost graph
    plt.xkcd() #changed
    plt.plot(range(1000), costs)
    plt.title("Cost Variation")

    plt.show()
with sess.as_default():
    print("Accuracy: %.2f" %accuracy.eval({X: test_x, Y: [t for t in test_y.as_matrix()]}))
