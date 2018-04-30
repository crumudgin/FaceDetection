import numpy as np
import scipy as sc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

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
    def __init__(self, learningRate, dataSize, classifiers, outputSize):
        self.learningRate = learningRate
        self.batchSize = 100
        self.minimize = None
        self.finalOutputSize = outputSize
        self.y = tf.placeholder(tf.int64, [self.batchSize], name='y')
        self.x = tf.placeholder(tf.float32, [self.batchSize, dataSize[0], dataSize[1], 3], name='x')
        self.keepProb = tf.placeholder(tf.float32, name='keepProb')
        self.previousLayer = None
        self.occurences = np.zeros((classifiers))
        self.sums = np.zeros((classifiers, 128))

    def activationSummary(self, var):
        pass
        # with tf.name_scope('summaries'):
        #     mean = tf.reduce_mean(var)
        #     tf.summary.scalar('mean', mean)
        #     with tf.name_scope('stddev'):
        #       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #     tf.summary.scalar('stddev', stddev)
        #     # tf.summary.scalar('max', tf.reduce_max(var))
        #     # tf.summary.scalar('min', tf.reduce_min(var))
        #     tf.summary.histogram('histogram', var)

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
        with tf.name_scope("conv_layer"):
            w = tf.Variable(tf.random_uniform(convFiltShape, minval=-1, maxval=1), name=name+'_W')
            # self.activationSummary(w)
            bias = tf.Variable(tf.random_uniform([numFilters], minval=-1, maxval=1), name=name+'_b')
            # self.activationSummary(bias)

            outLayer = tf.nn.conv2d(self.previousLayer, w, [1, 1, 1, 1], padding='SAME')
            # self.activationSummary(outLayer)

            outLayer += bias
            # self.activationSummary(outLayer)

            # outLayer = nonLiniarity(outLayer)
            # self.activationSummary(outLayer)
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
        with tf.name_scope("dense_layer"):
            wd = tf.Variable(tf.random_uniform([x, z]), name='wd' + name)
            # self.activationSummary(wd)
            bd = tf.Variable(tf.random_uniform([z]), name='bd' + name)
            # self.activationSummary(bd)
            dense_layer = tf.matmul(self.previousLayer, wd) + bd
            # self.activationSummary(dense_layer)
        self.previousLayer = nonLiniarity(dense_layer)
        return dense_layer

    def batchNorm(self, inputs):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
            inputs=inputs, axis=3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, fused=True)

    def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, data_format)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)

    def resBlock(self, chanels, filters, name, skip=True):
        shortcut = tf.identity(self.previousLayer)
        self.previousLayer = self.batchNorm(self.previousLayer)
        self.previousLayer = tf.nn.relu(self.previousLayer)
        # self.previousLayer = tf.nn.batch_normalization(self.previousLayer, 2, 2, 2, 2, 1e-5)
        self.createNewConvLayer(filters, filters, [3, 3], "conv_1_%s" %name)
        self.previousLayer = self.batchNorm(self.previousLayer)
        self.previousLayer = tf.nn.relu(self.previousLayer)
        self.createNewConvLayer(filters, filters, [3, 3], "conv_2_%s" %name)
        self.previousLayer = self.batchNorm(self.previousLayer)
        # self.createNewConvLayer(filters, filters, [1, 1], "conv_3_%s" %name)
        if skip:
            self.previousLayer += shortcut
        self.previousLayer = tf.nn.relu(self.previousLayer)





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
    def setNetwork(self, numOfConvs, numOfBlocks, numOfConnects, xSize):
        filters = 32
        inputChannels = 3
        counter = 1
        self.previousLayer = self.x
        self.createNewConvLayer(inputChannels, filters, [7,7], "conv_1")
        self.previousLayer = tf.nn.relu(self.previousLayer)
        # self.previousLayer = tf.nn.lrn(self.previousLayer)
        self.createPoolLayer([3, 3])
        self.resBlock(inputChannels, filters, "1")
        self.resBlock(inputChannels, filters, "2")
        self.resBlock(inputChannels, filters, "3")
        self.resBlock(inputChannels, filters, "4")
        # self.resBlock(inputChannels, filters, "5")
        # self.resBlock(filters * 2, filters * 2, "6")
        self.createPoolLayer([7, 7], tf.nn.avg_pool)
        # for i in range(0, numOfBlocks):
        #     for j in range(0, numOfConvs):
        #         if counter == 1:
        #             self.previousLayer = self.x
        #         self.createNewConvLayer(inputChannels, filters, [5, 5], str(counter))
        #         counter += 1
        #         inputChannels = filters
        #         filters *= 2
        #     self.createPoolLayer([2, 2])
        #     # self.previousLayer = tf.nn.lrn(self.previousLayer)
        # print(self.previousLayer.shape)
        shape = self.previousLayer.shape
        print(self.previousLayer.shape)
        self.previousLayer = tf.reshape(self.previousLayer, [-1, shape[1] * shape[2] * shape[3]])
        print(self.previousLayer.shape)
        ySize = 128
        finalOut = self.previousLayer
        # for i in range(0, numOfConnects-1):
        #     print(xSize)
        #     finalOut = self.createConnectedLayer((401408), ySize, tf.nn.relu, str(counter))
        #     xSize = ySize
        # finalOut = self.createConnectedLayer(100352, 128, tf.nn.relu, str(counter))
        # finalOut = tf.layers.dropout(inputs=finalOut, rate=self.keepProb)
        finalOut = self.createConnectedLayer(100352, self.finalOutputSize, tf.nn.relu, str(counter))
        finalOut = tf.nn.l2_normalize(finalOut)

        self.finalOut = finalOut

    
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

    def batchAll(self, labels, embedings, margin, squared=False):
        dist = self.pairwiseDist(embedings, squared=squared)
        anchorPosDist = tf.expand_dims(dist, 2)
        anchorNegDist = tf.expand_dims(dist, 1)
        loss = anchorPosDist - anchorNegDist + margin
        mask = self.getTripletMask(labels)
        mask = tf.to_float(mask)
        loss = tf.multiply(mask, loss)
        loss = tf.maximum(loss, 0.0)
        triplets = tf.to_float(tf.greater(loss, 1e-16))
        posTriplets = tf.reduce_sum(triplets)
        negTriplets = tf.reduce_sum(mask)
        ratio = posTriplets/(negTriplets + 1e-16)
        loss = tf.reduce_sum(loss) / (posTriplets + 1e-16)
        return loss, ratio


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


    def readTensor(self, dataType, ranomise):
        features = {"%s/label" %dataType: tf.FixedLenFeature([], tf.int64),
               "%s/image" %dataType: tf.FixedLenFeature([], tf.string)}

        queue = tf.train.string_input_producer(["%s.tfrecords" %dataType], num_epochs=None, shuffle=ranomise)
        reader = tf.TFRecordReader()
        _, example = reader.read(queue)
        features = tf.parse_single_example(example, features = features)
        label = features["%s/label" %dataType]
        image = features["%s/image" %dataType]
        if ranomise:
            imagesBatch, labelsBatch = tf.train.shuffle_batch(
                [image, label], batch_size=self.batchSize,
                capacity=self.batchSize*8,
                min_after_dequeue=self.batchSize,
                num_threads=8,
                allow_smaller_final_batch=True)
        else:
            imagesBatch, labelsBatch = tf.train.batch(
                [image, label], 
                batch_size=self.batchSize,
                num_threads=1,
                capacity=self.batchSize*8,
                allow_smaller_final_batch=True)
        image = tf.decode_raw(imagesBatch, tf.int8)
        image = tf.reshape(image, [-1, 224, 224, 3])
        return labelsBatch, image

    def calcAverage(self, image, label):
        self.occurences[label] += 1
        self.sums[label] += image


    def train(self, sess, totalBatch, testBatch, epochs):
        labelsBatch, imagesBatch = self.readTensor("train", True)
        testLabels, testImages = self.readTensor("train", False)
        self.setNetwork(1, 3, 2, 56*56*64)
        saver = tf.train.Saver()
        # loss, err = self.batchAll(self.y, self.finalOut, 2.0, True)
        # with tf.name_scope("loss"):
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(self.y, self.finalOut)
            # tf.summary.scalar('mean', tf.reduce_mean(loss))
        optimiser = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(loss)

        # merged = tf.summary.merge_all()
        trainWriter = tf.summary.FileWriter("/tb/summary/train", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, "/models/face_model_v6.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for j in range(epochs):
            print("starting epoch: %d of %d" %(j+1,epochs))
            cost = 0
            for i in range(totalBatch):
                img, label = sess.run([imagesBatch, labelsBatch])
                # print(label)
                if i % 100 == 0:
                    print("starting batch: %d of %d" %(i, totalBatch))
                _, c = sess.run([optimiser, loss], feed_dict={self.x: img, self.y: label, self.keepProb: .8})
                cost += c
            #     print(error)
            print(cost/totalBatch)
                    # trainWriter.add_summary(summary, i)
        saver.save(sess, "/models/face_model_v7.ckpt")
        wins = 0
        embeds = []
        for i in range(testBatch):
            if i % 100 == 0:
                print("calculating avreges for batch: %d of %d" %(i, testBatch))
            img, label = sess.run([testImages, testLabels])
            # print(label[i])
            embedings = sess.run(self.finalOut, feed_dict={self.x: img, self.y: label, self.keepProb: 1.0})
            # print(embedings[i])
            for k in range(self.batchSize):
                self.calcAverage(embedings[k], label[k])
            embeds.append(embedings)
        for k in range(self.sums.shape[0]):
            if self.occurences[k] == 0:
                self.sums[k] = 0
            else:
                self.sums[k] = self.sums[k] / self.occurences[k]
                # print(k)
                # print(self.occurences[k])
                # print(self.sums[k])
        for i in range(testBatch):
            if i % 100 == 0:
                print("testing batch: %d of %d" %(i, testBatch))
            img, label = sess.run([testImages, testLabels])
            # embedings = sess.run(self.finalOut, feed_dict={self.x: img, self.y: label, self.keepProb: 1.0})
            embs = tf.placeholder("float32", shape=(128,))
            dist = tf.reduce_sum(tf.square(self.sums - embs), 1)
            for k in range(self.batchSize):
                diff = None
                diffName = None
                diffs = sess.run(dist, feed_dict={embs:embeds[i][k]})
                for j in range(diffs.shape[0]):
                    currDiff = diffs[j]
                    if diff is None or currDiff < diff:
                        diff = currDiff
                        diffName = j
                if diffName == label[k]:
                    print("Correct: ", diffName, label[k])
                    wins += 1
                else:
                    print("Incorrect: ", diffName, label[k])
        print(wins/(testBatch*self.batchSize))


        coord.request_stop()
        coord.join(threads)