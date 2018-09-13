#!/usr/bin/python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2018 Surajit Saikia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


images = np.loadtxt('./fashionmnist/fashion-mnist_train.csv', delimiter = ',', skiprows=1)[:,1:]


""" we need to define the encoder and decoder part"""

#ENCODER
nodes_input_layer = 784
nodes_first_hidden = 32

#DECODER
nodes_second_hidden = 32
nodes_output_layer = 784






# We now define the weight matrixes
h1_weights = {
                      'weights':tf.Variable(tf.random_normal([nodes_input_layer,nodes_first_hidden])),
                      'bias':tf.Variable(tf.random_normal([nodes_first_hidden]))  }

# second hidden layer has 32*32 weights and 32 biases
h2_weights = {
                      'weights':tf.Variable(tf.random_normal([nodes_first_hidden, nodes_second_hidden])),
                      'bias':tf.Variable(tf.random_normal([nodes_second_hidden]))  }

# second hidden layer has 32*784 weights and 784 biases
output_weights = {
                    'weights':tf.Variable(tf.random_normal([nodes_second_hidden,nodes_output_layer])),               
                     'bias':tf.Variable(tf.random_normal([nodes_output_layer])) }





# the shape of the image is 28x28=784
input_layer = tf.placeholder('float', [None, 784])

layer_1 = tf.nn.sigmoid( tf.add(tf.matmul(input_layer,h1_weights['weights']),h1_weights['bias']))

layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,h2_weights['weights']), h2_weights['bias']))

output_layer = tf.matmul(layer_2,output_weights['weights']) + output_weights['bias']

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float', [None, 784])



# define the  cost function and optimizer
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))
learn_rate = 0.05   
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)





batch_size = 200
n_epochs =1000
size_data = 60000


#Intialize the session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())



    print("started TRAINING")
    for n in range(n_epochs):

        #print("epoch",epoch)
        epoch_loss = 0    # initializing error as 0

        for i in range(int(size_data/batch_size)):

            epoch_x = images[ i*batch_size : (i+1)*batch_size ]

            _, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x, output_true: epoch_x})

            epoch_loss += c


        print('epoch_number', n, ':', n_epochs, 'loss:',epoch_loss)



    # pick an image (i.e. the hundredth sample in my case)
    sample = images[100]
    # run it though the autoencoder
    output_sample = sess.run(output_layer,\
                   feed_dict={input_layer:[sample]})

    # run it though just the encoder
    encoded_any_image = sess.run(layer_1,\
                   feed_dict={input_layer:[sample]})



    plt.imshow(np.reshape(output_sample[0],(28,28)), cmap='Greys')
    plt.show()

