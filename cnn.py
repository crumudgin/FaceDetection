import numpy as np
import scipy as sc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class CNN():

    """
    Name: __init__
    Description: Initiates all the tensor variables and placeholders
                 while also setting up the framework for minimization
                 and the passing of data to and from networks
    Parameters: float learningRate - the value that determines the rate
                                     at which the network learns at
                tuple dataSize - the shape of the input data to the network
                int outputSize - the shape of the output for the network
    """
    def __init__(self, learningRate, dataSize, outputSize):
        self.learningRate = learningRate
        self.batchSize = 50
        self.minimize = None
        self.finalOutputSize = outputSize
        self.y = tf.placeholder(tf.float32, [None, outputSize], name='y')
        self.x = tf.placeholder(tf.float32, [None, dataSize[0] * dataSize[1]], name='x')
        self.x_shaped = tf.reshape(self.x, [-1, dataSize[0], dataSize[1], 1])
        self.previousLayer = self.x_shaped

    """
    Name: createNewConvLayer
    Description: Creates a "Convolution layer" of the network that will
                 perform a weighted convolution over the input layer.
                 The weights are determined through the overall cost
                 function and adjusted during training
    Parameters: int numInputChannels - the depth of the input "image" or layer
                int numFilters - the number of convolutions over the "image"
                (int, int) filterShape - the shape of the convolution
                string name - the name that is to be associated with the layer
                func nonLiniarity - a function (defaulted to relu) to be applied
                              to the convolved output of the layer
    returns: a matrix consisting of the relu applied to the convolved
             values
    """
    def createNewConvLayer(self, numInputChannels, numFilters, filterShape, name, nonLiniarity=tf.nn.relu):
        convFiltShape = [filterShape[0], filterShape[1], numInputChannels, numFilters]
        w = tf.Variable(tf.truncated_normal(convFiltShape, stddev=.03), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([numFilters]), name=name+'_b')

        outLayer = tf.nn.conv2d(self.previousLayer, w, [1, 1, 1, 1], padding='SAME')

        outLayer += bias

        outLayer = nonLiniarity(outLayer)
        self.previousLayer = outLayer
        return outLayer

    """
    Name: createPoolLayer
    Description: Creates a "Pooling layer" of the network that will
                 perform the designated pool over the previous layer
    Parameters: (int, int) poolShape - the shape of the pool boundaries
                func pool - the specific pool function to be applied
                            to the previous layer
    Returns: a matrix consisting of the pooled values
    """
    def createPoolLayer(self, poolShape, pool=tf.nn.max_pool):
        ksize = [1, poolShape[0], poolShape[1], 1]
        strides = [1, 2, 2, 1]
        outLayer = pool(self.previousLayer, ksize=ksize, strides=strides, padding='SAME')
        self.previousLayer = outLayer
        return outLayer

    """
    Name: createConnectedLayer
    Description: Creates a "Fully Connected layer" of the network that
                 will intemperate the output of the pooling layer to 
                 establish patterns and aid in classification of the
                 image
    Parameters: int x - the x value of the input layer's shape
                int y - the y value of the input layer's shape
                func nonLiniarity - the nonlinear function to
                                    be applied to the output
                                    of the layer
                string name - the name to be assigned to the layer
    Returns: the output of the layer without the nonliniarity function
             applied to it
    """
    def createConnectedLayer(self, x, z, nonLiniarity, name):
        wd = tf.Variable(tf.truncated_normal([x, z], stddev=.03), name='wd' + name)
        bd = tf.Variable(tf.truncated_normal([z], stddev=0.01), name='bd' + name)
        dense_layer = tf.matmul(self.previousLayer, wd) + bd
        self.previousLayer = nonLiniarity(dense_layer)
        return dense_layer

    """
    Name: setNetwork
    Description: Initializes the structure of the network in the following pattern
                 convolution -> pooling -> fully connected
    Parameters: int numOfConvs - the number of convolution layers in between each
                                 pooling layer
                int numOfBlocks - the number of pooling layers
                int numOfConnects - the number of fully connected layers at the end
                                    of the network
    Returns: the cost function
    """
    def setNetwork(self, numOfConvs, numOfBlocks, numOfConnects, dataSize):
        filters = 32
        inputChannels = 1
        counter = 1
        for i in range(0, numOfBlocks):
            for j in range(0, numOfConvs):
                self.createNewConvLayer(inputChannels, filters, [5, 5], str(counter))
                counter += 1
                inputChannels = filters
                filters *= 2
            self.createPoolLayer([2, 2])
        xSize = dataSize*filters//2
        ySize = 1000
        print(self.previousLayer.shape)
        self.previousLayer = tf.reshape(self.previousLayer, [-1, xSize])
        for i in range(0, numOfConnects-1):
            finalOut = self.createConnectedLayer(xSize, ySize, tf.nn.relu, str(counter))
            xSize = ySize
        finalOut = self.createConnectedLayer(xSize, self.finalOutputSize, tf.nn.relu, str(counter))
        finalOut = tf.nn.softmax(finalOut)
        self.previousLayer = finalOut
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=finalOut, labels=self.y))
        self.minimize = cross_entropy
        return cross_entropy


    """
    Name: train
    Description: performs training on the network with the goal of minimizing the cost
                 function
    Parameters: int epochs - the number of rounds of training the network should attempt
    """
    def train(self, epochs, data, labels, testData, testLabels):
        if self.minimize is None:
            print("you need to set the network first")
            return
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.minimize)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.previousLayer, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        initOptimiser = tf.global_variables_initializer()

        with tf.Session() as sess:
            # initialise the variables
            sess.run(initOptimiser)
            total_batch = int(len(labels) / self.batchSize)
            for epoch in range(epochs):
                avg_cost = 0
                currBatch = 0
                batch = self.batchSize
                for i in range(total_batch):
                    batch_x = data[currBatch:batch]
                    batch_y = labels[currBatch:batch]
                    currBatch = batch
                    batch += self.batchSize
                    if batch > len(labels):
                        batch = len(labels)
                    _, c = sess.run([optimiser, self.minimize], 
                                    feed_dict={self.x: batch_x, self.y: batch_y})
                    avg_cost += c / total_batch
                test_acc = sess.run(accuracy, 
                               feed_dict={self.x: testData, self.y: testLabels})
                print("EPOCH #%s complete accuacy at %s" %(epoch, test_acc))

            print("\nTraining complete!")
            print(sess.run(accuracy, feed_dict={self.x: testData, self.y: testLabels}))

    def run(self, data):
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.minimize)
        prediction = self.previousLayer
        
        initOptimiser = tf.global_variables_initializer()

        with tf.Session() as sess:
            # initialise the variables
            sess.run(initOptimiser)
                
            pred = sess.run(prediction,
                            feed_dict={self.x: data})
            print("Prediction: ", pred)