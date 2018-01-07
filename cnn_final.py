import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import scipy.io 
import csv
from sklearn.metrics import confusion_matrix

mat = scipy.io.loadmat('imdb.mat')
arr = mat['images']['data'][0][0]
original_labels = mat['images']['labels'][0][0][0]
original_labels_1 = original_labels[45000:50000,...]

x_total = arr.reshape([3072, 60000])
x_total = x_total.T
X = x_total[0:45000,...]
x_valid = x_total[45000:50000,...]
x_test1 = x_total[50000:52500,...]
x_test2 = x_total[52500:55000,...]
x_test3 = x_total[55000:57500,...]
x_test4 = x_total[57500:60000,...]



n = len(original_labels)
original_labels = [int(tmp-1) for tmp in original_labels]
a = np.array(original_labels)
Y = np.zeros((n, 100))
Y[np.arange(n), original_labels] = 1

y_valid = Y[45000:50000,...]
Y = Y[0:45000,...]
# Y_test = to_categorical(Y_test, 100)

def weight_variable(shape):
  initial = tf.zeros(shape)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# print(X[0].shape)
# plt.figure()
# plt.imshow(X[1])
# plt.show()
#################################3
x = tf.placeholder(tf.float32, [None, 3072])
y_ = tf.placeholder(tf.float32, [None, 100])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,32,32,3])

h_conv1 = tf.nn.lrn((conv2d(x_image, W_conv1) + b_conv1), 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_relu1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(h_relu1)

#################################3
W_conv2 = weight_variable([7, 7, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.lrn((conv2d(h_pool1, W_conv2) + b_conv2), 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_relu2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_relu2)


#################################3
W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.lrn((conv2d(h_pool2, W_conv3) + b_conv3), 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_relu3 = tf.nn.relu(h_conv3)
h_pool3 = max_pool_2x2(h_relu3)

#################################3
keep_prob = tf.placeholder(tf.float32)
W_fc1 = weight_variable([4 * 4 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#################################3

# W_fc2 = weight_variable([1024, 512])
# b_fc2 = bias_variable([512])

# h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


#################################3

W_fc3 = weight_variable([1024, 100])
b_fc3 = bias_variable([100])

y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3
h_fc3_drop = tf.nn.dropout(y_conv, keep_prob)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay = 1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 2
dropout = 1
error_t = []
error =[]
for k in range(epochs):
	for i in range(449):
	  j = i
	  X1=X[(j)*100:(j+1)*100,...]
	  Y1=Y[(j)*100:(j+1)*100,...]

	  train_step.run(feed_dict={x: X1, y_: Y1, keep_prob: dropout})
	  if i%100 == 0:
	  	train_accuracy = accuracy.eval(feed_dict={x:X1, y_: Y1, keep_prob: 1.0})
	  	print("epoch: %d, step: %d, training accuracy %g"%(k, i*100, train_accuracy))
	error_t+=[ 1-(accuracy.eval(feed_dict={x:X1, y_: Y1, keep_prob: 1.0}))]
	error+=[ 1-(accuracy.eval(feed_dict={x:x_valid, y_: y_valid, keep_prob: 1.0}))]
	print("valid accuracy %g"%accuracy.eval(feed_dict={x:x_valid, y_: y_valid, keep_prob: 1.0}))
# valid_accuracy = accuracy.eval(feed_dict={x:x_valid, y_: y_valid, keep_prob: 1.0})
# print("valid accuracy %g"%valid_accuracy)
plt.plot(range(len(error)),error,'r--' , label="validation error")
plt.plot(range(len(error_t)),error_t,'b--', label="train error")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

prediction = tf.argmax(y_conv, 1)
valid_accuracy = sess.run(prediction, feed_dict={x: x_valid, keep_prob: 1.0})

conf_matrix = confusion_matrix(original_labels_1, valid_accuracy)
plt.matshow(conf_matrix)
plt.title('Training Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('conf_matrix_train.png')
plt.close()
# tf.contrib.metrics.confusion_matrix(y_valid,valid_accuracy num_classes=None,dtype=tf.int32,name=None,weights=None)



#test1
prediction = tf.argmax(y_conv, 1)
test_predicted = sess.run(prediction, feed_dict={x: x_test1, keep_prob: 1.0})

csv_content = [["ID", "Label"]]
for ind,data in enumerate(test_predicted):
	csv_content.append([ind+1, data+1])
with open("cifar_prediction1.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(csv_content)

#test2
prediction = tf.argmax(y_conv, 1)
test_predicted = sess.run(prediction, feed_dict={x: x_test2, keep_prob: 1.0})
csv_content = [["ID", "Label"]]
for ind,data in enumerate(test_predicted):
	csv_content.append([ind+2500+1, data+1])
with open("cifar_prediction2.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(csv_content)

#test3
prediction = tf.argmax(y_conv, 1)
test_predicted = sess.run(prediction, feed_dict={x: x_test2, keep_prob: 1.0})
csv_content = [["ID", "Label"]]
for ind,data in enumerate(test_predicted):
	csv_content.append([ind+5000+1, data+1])
with open("cifar_prediction3.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(csv_content)


#test4
prediction = tf.argmax(y_conv, 1)
test_predicted = sess.run(prediction, feed_dict={x: x_test4, keep_prob: 1.0})
csv_content = [["ID", "Label"]]
for ind,data in enumerate(test_predicted):
	csv_content.append([ind+7500+1, data+1])
with open("cifar_prediction4.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(csv_content)



fout=open("out.csv","a")
# first file:
for line in open("cifar_prediction1.csv"):
    fout.write(line)
# now the rest:    
for num in range(2,5):
    f = open("cifar_prediction"+str(num)+".csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()


