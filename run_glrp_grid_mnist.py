#!python

"""
Running the GLRP on GCNN model trained on MNIST data. Digits are graph signals on 8-nearest neighbor graph.
The model can be retrained uncommenting a part of the code.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.examples.tutorials.mnist import input_data

from components import glrp_scipy, visualize_mnist
from lib import models, graph, coarsening

import time

COARSENING_LEVELS = 4  # to satisfy pooling of size 4 two times we need 4 level
DIR_DATA = "./data/mnist"
METRIC = 'euclidean'
NUMBER_EDGES = 8
M = 28  # size of the digit's picture side
FEATURE_NUM = M * M
EPS = 1e-7  # for adjacency matrix


if __name__ == "__main__":
    # !!!
    # creating the adjacency matrix with all the non-zero weights equal to 1
    z = graph.grid(M)
    dist, idx = graph.distance_sklearn_metrics(z, k=NUMBER_EDGES, metric=METRIC)
    A = graph.adjacency(dist, idx)

    A[A > EPS] = 1

    graphs, perm = coarsening.coarsen(A, levels=COARSENING_LEVELS, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    mnist = input_data.read_data_sets(DIR_DATA, one_hot=False)

    train_data = mnist.train.images.astype(np.float32)
    val_data = mnist.validation.images.astype(np.float32)
    test_data = mnist.test.images.astype(np.float32)
    train_labels = mnist.train.labels
    val_labels = mnist.validation.labels
    test_labels = mnist.test.labels

    train_data = coarsening.perm_data(train_data, perm)
    val_data = coarsening.perm_data(val_data, perm)
    test_data = coarsening.perm_data(test_data, perm)

    common = {}
    common['dir_name'] = 'mnist_grid_ones/'
    common['num_epochs'] = 30
    common['batch_size'] = 100
    common['decay_steps'] = mnist.train.num_examples / common['batch_size']
    common['eval_frequency'] = 30 * common['num_epochs']
    common['brelu'] = 'b1relu'
    common['pool'] = 'mpool1'
    C = max(mnist.train.labels) + 1  # number of classes

    common['regularization'] = 5e-4
    common['dropout'] = 0.5
    common['learning_rate'] = 0.03
    common['decay_rate'] = 0.95
    common['momentum'] = 0.9
    common['F'] = [32, 64]
    common['K'] = [25, 25]
    common['p'] = [4, 4]
    common['M'] = [512, C]

    name = 'cgconv_cgconv_softmax_momentum'
    params = common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'

    model = models.cgcnn(L, **params)

    # !!!
    # To train again uncomment this part
    # start = time.time()
    # accuracy, loss, t_step, trained_losses = model.fit(train_data, train_labels, val_data, val_labels)
    # end = time.time()

    probas_ = model.get_probabilities(test_data)
    f1 = 100 * f1_score(test_labels, np.argmax(probas_, axis=1), average='weighted')
    acc = 100 * accuracy_score(test_labels, np.argmax(probas_, axis=1))
    print("\n\tTest F1 weighted: ", f1)
    print("\tTest Accuraccy:", acc, "\n")

    data_to_test = val_data[0:common["batch_size"], ]
    probas_ = model.get_probabilities(data_to_test)
    labels_by_network = np.argmax(probas_, axis=1)
    labels_data_to_test = val_labels[0:common["batch_size"], ]
    I = np.eye(10)
    labels_hot_encoded = I[labels_by_network]

    glrp = glrp_scipy.GraphLayerwiseRelevancePropagation(model, data_to_test, labels_hot_encoded)
    rel = glrp.get_relevances()[-1]  # getting the relevances corresponding to the input layer

    data_to_test = coarsening.perm_data_back(data_to_test, perm, FEATURE_NUM)
    rel = coarsening.perm_data_back(rel, perm, FEATURE_NUM)

    results_dir = './figures/'

    start_example = 9
    end_example = 17

    visualize_mnist.plot_numbers(data_to_test[start_example:end_example, ], rel[start_example:end_example, ],
                                 labels_data_to_test[start_example:end_example, ],
                                 labels_by_network[start_example:end_example, ], results_dir)

    # start_example = 9
    # end_example = 17
    for i in range(start_example, end_example):  # 9, 17
        heatmap = visualize_mnist.get_heatmap(rel[i,])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(heatmap, cmap='Reds', interpolation='bilinear')
        fig.savefig('{0}LRP_w^+_correct_label_index{1}_{2}.png'.format(results_dir, str(i), str(val_labels[i])))
