import numpy as np
import scipy as sc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
        self.batchSize = 800
        self.minimize = None
        self.finalOutputSize = outputSize
        self.y = tf.placeholder(tf.float32, [self.batchSize], name='y')
        self.x = tf.placeholder(tf.float32, [None, dataSize[0] * dataSize[1] * 1], name='x')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        self.x_shaped = tf.reshape(self.x, [-1, dataSize[0], dataSize[1], 1])
        self.previousLayer = self.x_shaped
        self.anchor = tf.placeholder(tf.float32, [self.batchSize, 128], name='anchor')
        self.negative = tf.placeholder(tf.float32, [self.batchSize, 128], name='negative')
        self.batchSpot = 0
        self.outputs = []

    def activationSummary(self, x):
        tf.summary.histogram(x.op.name, x)
        tf.summary.scalar(x.op.name, tf.nn.zero_fraction(x))

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
        w = tf.Variable(tf.random_uniform(convFiltShape), name=name+'_W')
        bias = tf.Variable(tf.random_uniform([numFilters]), name=name+'_b')

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
        wd = tf.Variable(tf.random_uniform([x, z]), name='wd' + name)
        bd = tf.Variable(tf.random_uniform([z]), name='bd' + name)
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
    def setNetwork(self, numOfConvs, numOfBlocks, numOfConnects, dataSize, xSize):
        filters = 64
        inputChannels = 1
        counter = 1
        for i in range(0, numOfBlocks):
            for j in range(0, numOfConvs):
                if counter == 1:
                    self.previousLayer = self.x_shaped
                self.createNewConvLayer(inputChannels, filters, [5, 5], str(counter))
                counter += 1
                inputChannels = filters
                filters *= 2
            self.createPoolLayer([2, 2])
        print(self.previousLayer.shape)
        self.previousLayer = tf.reshape(self.previousLayer, [-1, xSize ])
        print(self.previousLayer.shape)
        ySize = 128
        for i in range(0, numOfConnects-1):
            print(xSize)
            finalOut = self.createConnectedLayer(xSize, ySize, tf.nn.relu, str(counter))
            xSize = ySize
        finalOut = tf.layers.dropout(inputs=finalOut, rate=self.keepProb)
        finalOut = self.createConnectedLayer(xSize, self.finalOutputSize, tf.nn.relu, str(counter))
        finalOut = tf.nn.l2_normalize(finalOut)
        # finalOut = tf.nn.softmax(finalOut)

        self.finalOut = finalOut
        # print(finalOut.shape)
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=finalOut, labels=self.y))
        # self.minimize = cross_entropy
        # self.outputs.append(finalOut)
        # return cross_entropy

    
    def pairwiseDist(self, embedings, squared=False):
        dot = tf.matmul(embedings, tf.transpose(embedings))
        squaredNorm = tf.diag_part(dot)
        distance = tf.expand_dims(squaredNorm, 0) -2.0 * dot + tf.expand_dims(squaredNorm, 1)
        distance = tf.maximum(distance, 0.0)
        if not squared:
            mask = tf.to_float(tf.equal(distance, 0.0))
            distance = distance+mask*1e-16
            distance = tf.sqrt(distance)
            distance = distance * (1.0 - mask)
        return distance


    def getAnchorPositiveMask(self, labels):
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return mask


    def getAnchorNegativeMask(self, labels):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        mask = tf.logical_not(labels_equal)

        return mask


    def getTripletMask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask


    def batchHard(self, embedings, labels, margin=1.0, squared=False):
        dist = self.pairwiseDist(embedings, squared=squared)
        maskAnchorPositive = self.getAnchorPositiveMask(labels)
        maskAnchorPositive = tf.to_float(maskAnchorPositive)
        positiveAnchorDist = tf.multiply(maskAnchorPositive, dist)
        hardestPositive = tf.reduce_max(positiveAnchorDist, axis=1, keepdims=True)

        maskAnchorNegative = self.getAnchorNegativeMask(labels)
        maskAnchorNegative = tf.to_float(maskAnchorNegative)
        maxAnchorNegative = tf.reduce_max(dist, axis=1, keepdims=True)
        anchorNegativeDist = dist + maxAnchorNegative * (1.0 - maxAnchorNegative)
        hardestNegative = tf.reduce_max(anchorNegativeDist, axis=1, keepdims=True)

        trippletLoss = tf.maximum(hardestPositive - hardestNegative + margin, 0.0)
        return tf.reduce_mean(trippletLoss)

    def tripletTrain(self, epochs, data, labels, testData, testLabels, anchors, anchorLabels, sess):
        print(len(testData), self.batchSize)
        saver = tf.train.Saver()
        total_batch = int(len(labels) / self.batchSize)
        prediction = self.finalOut
        test_batch = int(len(testData) / self.batchSize)
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(self.y, self.finalOut)
        # loss = self.batchHard(self.finalOut, self.y, 1.0)
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(loss)
        initOptimiser = tf.global_variables_initializer()
        sess.run(initOptimiser)
        # accuracy = self.trippletAccuracy(prediction)
        # saver.restore(sess, "/models/face_model.ckpt")
        batch = self.batchSize
        currBatch = 0
        print("initial testing")
        # self.loadFaces(sess, anchors, anchorLabels)
        # self.tripletError(sess, testData, testLabels)
        currBatch = 0
        batch = self.batchSize
        err = 0
        for i in range(test_batch):
            if i % 100 == 0:
                print("starting test batch %d of %d" %(i, total_batch))
            batch_x = testData[currBatch:batch]
            batch_y = testLabels[currBatch:batch]
            currBatch = batch
            batch += self.batchSize
            err += sess.run(loss, feed_dict={self.x:batch_x, self.y: batch_y, self.keepProb: .4})
        print(1 - err/test_batch)
        for epoch in range(epochs):
            print("starting epoch %d" %epoch)
            avg_cost = 0
            currBatch = 0
            batch = self.batchSize
            for i in range(total_batch):
                # if i % 100 == 0:
                #     print("starting batch %d of %d" %(i, total_batch))
                batch_x = data[currBatch:batch]
                batch_y = labels[currBatch:batch]
                currBatch = batch
                batch += self.batchSize
                if batch > len(labels):
                    batch = len(labels)
                _, c = sess.run([optimiser, loss], 
                    feed_dict={self.x:batch_x, self.y: batch_y, self.keepProb: .4})
                # print(c)
        currBatch = 0
        batch = self.batchSize
        err = 0
        for i in range(test_batch):
            if i % 100 == 0:
                print("starting test batch %d of %d" %(i, total_batch))
            batch_x = testData[currBatch:batch]
            batch_y = testLabels[currBatch:batch]
            currBatch = batch
            batch += self.batchSize
            err += sess.run(loss, feed_dict={self.x:batch_x, self.y: batch_y, self.keepProb: .4})
        print(1 - err/test_batch)
                

        print("\nTraining complete!")
        saver.save(sess, "/models/face_model.ckpt")

    def trippletRun(self, sess, image, prediction):
        initOptimiser = tf.global_variables_initializer()
        sess.run(initOptimiser)
        saver = tf.train.Saver()
        # saver.restore(sess, "/models/new_model.ckpt")
        pred = sess.run(prediction, feed_dict={self.x: image, self.keepProb: 1})
        diff = None
        diffName = None
        for i in range(len(self.names)):
            currDiff = tf.reduce_sum(tf.square(pred - self.faces[i]), 1)
            currDiff = sess.run(currDiff)
            if diff is None or currDiff < diff:
                print(currDiff)
                diff = currDiff
                diffName = self.names[i]
        print(diffName)



    def loadFaces(self, sess, examples, names):
        saver = tf.train.Saver()
        prediction = self.finalOut
        initOptimiser = tf.global_variables_initializer()
        sess.run(initOptimiser)
        saver.restore(sess, "/models/face_model.ckpt")
        self.faces = []
        self.names = names
        print(self.names)
        for face in examples:
            feature = sess.run(prediction, feed_dict={self.x: [face], self.keepProb: 1})
            self.faces.append(feature)

    def tripletError(self, sess, data, labels):
        prediction = self.finalOut
        initOptimiser = tf.global_variables_initializer()
        sess.run(initOptimiser)
        saver = tf.train.Saver()
        saver.restore(sess, "/models/face_model.ckpt")
        wins = 0
        features = sess.run(self.finalOut, feed_dict={self.x:data, self.keepProb: 1})
        # measure = tf.reduce_mean(tf.reduce_sum(tf.square(features - self.faces), 1))
        measure = tf.reduce_sum(tf.square(features - self.faces), 2)
        measure = sess.run(measure)
        for i in range(len(labels)):
            diff = 100
            diffName = None
            for name in range(len(self.names)):
                if measure[name, i] < diff:
                    diff = measure[name, i]
                    diffName = self.names[name]
            if diffName == labels[i]:
                # print("yup", diffName, labels[i])
                wins += 1
            # else:
            #     print("Nope", diffName, labels[i])
        print(wins/len(labels))

