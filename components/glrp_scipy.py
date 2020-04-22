# MIT License
#
# Copyright (c) 2020 Hryhorii Chereda
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

from tensorflow.python.ops import gen_nn_ops
import numpy as np
import tensorflow as tf
import scipy
from lib import graph
import time


class GraphLayerwiseRelevancePropagation:
    """
    Class encapsulates functionality for running layer-wise relevance propagation on Graph CNN.
    """

    def __init__(self, model, samples, labels):
        """
        Initialization of internals for relevance computations.
        :param model: gcnn model to LRP procedure is applied on
        :param samples: which samples to calculate relevance on, num of samples == models batch size
        :param labels: used as values for the gcnn output nodes to propagate relevance
        """
        self.epsilon = 1e-10  # for numerical stability

        start = time.time()
        print("\n\tCalculating Polynomials of Laplace Matrices...", end=" ")
        self.polynomials = [self.calc_Laplace_Polynom(lap, K=model.K[i]) for i, lap in enumerate(model.L)]
        end = time.time()
        print("Time: ", end - start, "\n")

        weights = model.get_weights()
        self.activations = model.activations

        self.model = model
        self.model.graph._unsafe_unfinalize()  # the computational graph of the model will be modified
        self.labels = labels
        self.samples = samples
        self.X = self.activations[0]  # getting the first
        # self.y = self.activations.pop(0)
        # I am getting the activation of the first, but not useful here, self.y will be assigned a placeholder

        self.ph_dropout = model.ph_dropout
        self.batch_size = self.X.shape[0]

        with self.model.graph.as_default():
            self.y = tf.placeholder(tf.float32, (self.batch_size, labels.shape[1]), 'labels_hot_encoded')

        self.act_weights = {}  # example in this dictionary "conv1": [weights, bias]

        for act in self.activations[1:]:
            w_and_b = []  # 2 element list of weight and bias of one layer.
            name = act.name.split('/')
            # print(name)
            for wt in weights:
                # print(wt.name)
                if name[0] == wt.name.split('/')[0]:
                    w_and_b.append(wt)
            if w_and_b and (name[0] not in self.act_weights):
                self.act_weights[name[0]] = w_and_b

        self.activations.reverse()

        # !!!
        # first convolutional layer filters
        self.filters_gc1 = []
        self.filtered_signal_gc1 = []

    def get_relevances(self):
        """
        Computes relevances based on input samples.
        :param rule: the propagation rule of the first layer
        :return: the list of relevances, corresponding to different layers of the gcnn.
        The last element of this list contains input relevances and has shape (batch_size, num_of_input_features)
        """

        # Backpropagate softmax value
        # relevances = [tf.nn.softmax(self.activations[0])*tf.cast(self.y, tf.float32)]

        # Backpropagate a value from given labels y
        relevances = [tf.cast(self.y, tf.float32)]

        loc_poly = [pol for pol in self.polynomials]
        loc_pooling = [p for p in self.model.p]
        print("\n    Relevance calculation:")
        for i in range(1, len(
                self.activations)):  # start from 1 (not 0) penultimate activations since relevances already contains logits.
            name = self.activations[i - 1].name.split('/')
            if 'logits' in name[0] or 'fc' in name[0]:
                print("\tFully connected:", name[0])
                relevances.append(self.prop_fc(name[0], self.activations[i], relevances[-1]))
            elif 'flatten' in name[0]:
                print("\tFlatten layer:", name[0])
                relevances.append(self.prop_flatten(self.activations[i], relevances[-1]))
                # print("\n")
            elif 'pooling' in name[1]:
                # TODO: incorporate pooling type and value into name
                print("\tPooling:", name[0] + " " + name[1])
                p = loc_pooling.pop()
                relevances.append(self.prop_max_pool(self.activations[i], relevances[-1], ksize=[1, p, 1, 1],
                                                     strides=[1, p, 1, 1]))
            elif 'conv' in name[0]:
                if len(loc_poly) > 1:
                    print("\tConvolution: ", name[0], "\n")
                    relevances.append(self.prop_gconv(name[0], self.activations[i], relevances[-1],
                                                      polynomials=loc_poly.pop()))
                else:
                    print("\tConvolution, the first layer:", name[0], "\n")
                    relevances.append(self.prop_gconv_first_conv_layer(name[0], self.activations[i], relevances[-1],
                                                                       polynomials=loc_poly.pop()))
            else:
                raise 'Error parsing layer'

        return relevances

    def prop_fc(self, name, activation, relevance):
        """Propagates relevances through fully connected layers."""
        w = self.act_weights[name][0]
        # b = self.act_weights[name][1]  # bias
        w_pos = tf.maximum(0.0, w)
        z = tf.matmul(activation, w_pos) + self.epsilon
        s = relevance / z
        c = tf.matmul(s, tf.transpose(w_pos))
        return c * activation

    def prop_flatten(self, activation, relevance):
        """Propagates relevances from the fully connected part to convolutional part."""
        shape = activation.get_shape().as_list()
        return tf.reshape(relevance, shape)

    def prop_max_pool(self, activation, relevance, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1]):
        """Propagates relevances through max pooling."""
        act = tf.expand_dims(activation, 3)  # N x M x F x 1
        z = tf.nn.max_pool(act, ksize, strides, padding='SAME') + self.epsilon
        with self.model.graph.as_default():
            rel = tf.expand_dims(relevance, 3)
        s = rel / z
        c = gen_nn_ops.max_pool_grad_v2(act, z, s, ksize, strides, padding='SAME')
        tmp = c * act
        return tf.squeeze(tmp, [3])

    def prop_gconv(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through Graph Convolutional Layers.
        All essential operations are in SCIPY.
        """
        start = time.time()
        w = self.act_weights[name][0]  # weight
        b = self.act_weights[name][1]  # bias
        # print("\nInside gconv")
        # print("weights of current gconv, w:", w)
        # activation
        N, M, Fin = activation.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(w.get_shape().as_list()[-1])

        K = int(w.get_shape().as_list()[0] / Fin)

        W = self.model._get_session().run(w)

        # The values of a TF tensor needed
        activation = self.run_tf_tensor(activation, samples=self.samples)

        # if relevance is a tf.tensor then run the session
        if tf.contrib.framework.is_tensor(relevance):
            relevance = self.run_tf_tensor(relevance, self.samples)

        W = np.reshape(W, (int(W.shape[0] / K), K, Fout))
        W = np.transpose(W, (1, 0, 2))  # K x Fin x Fout

        rel = np.zeros(shape=[N, M * Fin], dtype=np.float32)
        for i in range(0, Fout):
            w_pos = polynomials.dot(W[:, :, i])
            w_pos = np.maximum(0.0, w_pos)
            w_pos = np.reshape(w_pos, [M, M, Fin])
            w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            w_pos = np.reshape(w_pos, [M * Fin, M])
            activation = np.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = np.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = np.matmul(s, np.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()
        # #
        rel = np.reshape(rel, [N, M, Fin])
        print("\n\t" + name + ",", "relevance propagation time is: ", end - start)

        return rel

    def prop_gconv_first_conv_layer(self, name, activation, relevance, polynomials):
        """
        Perform relevance propagation through the first Graph Convolutional Layer.
        All the essential operations are in SCIPY.
        """
        start = time.time()

        w = self.act_weights[name][0]
        b = self.act_weights[name][1]  # bias
        N, M, Fin = activation.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        Fout = int(w.get_shape().as_list()[-1])

        K = int(w.get_shape().as_list()[0] / Fin)

        W = self.model._get_session().run(w)
        B = self.model._get_session().run(b)
        activation = self.run_tf_tensor(activation, samples=self.samples)

        # Need in Numpy values for SciPy
        if tf.contrib.framework.is_tensor(relevance):
            relevance = self.run_tf_tensor(relevance, self.samples)

        rel = np.zeros(shape=[N, M], dtype=np.float32)

        for i in range(0, Fout):
            w_pos = polynomials.dot(W[:, i])
            self.filters_gc1.append(np.reshape(w_pos, [M, M]))
            self.filtered_signal_gc1.append(np.expand_dims(np.matmul(self.samples, self.filters_gc1[-1]), axis=2))
            w_pos = np.maximum(0.0, w_pos)
            w_pos = np.reshape(w_pos, [M, M])
            # w_pos = np.transpose(w_pos, axes=[0, 2, 1])  # M x Fin x M
            # w_pos = np.reshape(w_pos, [M * Fin, M])

            # !!!
            # Z^+ rule
            activation = np.reshape(activation, [N, Fin * M])  # N x Fin*M
            z = np.matmul(activation, w_pos) + self.epsilon  # N x M
            s = relevance[:, :, i] / z  # N x M
            c = np.matmul(s, np.transpose(w_pos))  # N x M by transpose(M * Fin, M) = N x M * Fin
            rel += c * activation
        end = time.time()

        print("\n\t" + name + ",", "relevance propagation time is: ", end - start, "\n")
        return rel

    def calc_Laplace_Polynom(self, L, K):
        """
        Calculate Chebyshev polynoms based on Laplace matrix.
        :param L: Laplace matrix M*M
        :param K: Number of polynoms with degree from 0 to K-1
        :return: Chebyshev Polynoms in scipy.sparse.coo, shape (M*M, K)
        """
        # N, M, Fin = self.X.get_shape()
        M = int(L.shape[0])
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.

        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        polynomials = []
        if K > 1:
            # only rank 2 for sparse_tensor_dense_matmul
            T0 = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            I = scipy.sparse.identity(M, dtype=np.float32, format="csr")
            T1 = L
            # polynomials.extend([I, T1]) # the first matrix is I matrix
            polynomials = scipy.sparse.hstack([I.reshape(M * M, 1), T1.reshape(M * M, 1)])
        for k in range(2, K):
            T2 = 2 * L * T1 - T0  #
            polynomials = scipy.sparse.hstack([polynomials, T2.reshape(M * M, 1)])
            T0, T1 = T1, T2
        return polynomials

    def run_tf_tensor(self, tensor, samples):
        """
        Runs computational graph to get the Numpy values of the tensor.
        :param tensor:
        :param samples:
        :return:the value of a specific tensor based on the
        """
        # TODO: rewrite it as a concatenation of all the batches per each activation
        data = np.expand_dims(samples, axis=2)
        size = data.shape[0]
        # activations = [] # np.empty([size, len(self.activations) - 2]) # TODO: adjust the number of tensors
        # print(R.shape)
        # session =
        with self.model._get_session() as sess:
            for begin in range(0, size, self.batch_size):
                end = begin + self.batch_size
                end = min([end, size])
                batch_data = np.zeros((self.batch_size, data.shape[1], 1))
                tmp_data = data[begin:end, :, :]
                if type(tmp_data) is not np.ndarray:
                    tmp_data = tmp_data.toarray()  # convert sparse matrices
                batch_data[:end - begin] = tmp_data
                feed_dict = {self.X: batch_data, self.y: self.labels,
                             self.ph_dropout: 1}
                batch_tnsr = sess.run(tensor, feed_dict)

                tnsr = batch_tnsr
                # for ba in batch_activations:
                #     print("The whole batch:")
                #     print("ba:", ba.shape)
                #     print("ba, Gb:", ba.nbytes / (1024*1024*1024))
                #     activations.append(ba)

        return tnsr
