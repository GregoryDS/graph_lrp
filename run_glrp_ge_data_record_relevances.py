#!python

"""
Running the GLRP on GCNN model trained on gene expression data. Obtained relevances are written into the file .
The model can be retrained uncommenting a part of the code.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split

from components import data_handling, glrp_scipy
from lib import models, graph, coarsening

# from sklearn.model_selection import StratifiedKFold

import time

rndm_state = 7
np.random.seed(rndm_state)

if __name__ == "__main__":

    path_to_feature_val = "./data/GE_PPI/GEO_HG_PPI.csv"
    path_to_feature_graph = "./data/GE_PPI/HPRD_PPI.csv"
    path_to_labels = "./data/GE_PPI/labels_GEO_HG.csv"

    DP = data_handling.DataPreprocessor(path_to_feature_values=path_to_feature_val, path_to_feature_graph=path_to_feature_graph,
                                        path_to_labels=path_to_labels)
    X = DP.get_feature_values_as_np_array()  # gene expression
    A = csr_matrix(DP.get_adj_feature_graph_as_np_array().astype(np.float32))  # adjacency matrix of the PPI network
    y = DP.get_labels_as_np_array()  # labels

    print("GE data, X shape: ", X.shape)
    print("Labels, y shape: ", y.shape)
    print("PPI network adjacency matrix, A shape: ", A.shape)

    X_train_unnorm, X_test_unnorm, y_train, y_test = train_test_split(X, y, test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Need to know which patients got into train and test subsets
    _, _, patient_indexes_train, patient_indexes_test = train_test_split(X, DP.labels.columns.values.tolist(), test_size=0.10,
                                                                      stratify=y, random_state=rndm_state)

    # Data frame with test patients and corresponding ground truth labels
    patient_ind_test_df = pd.DataFrame(data={"Patient ID": patient_indexes_test, "label": y_test})

    # !!!
    # Making data lying in the interval [0, 8.35]
    X_train = X_train_unnorm - np.min(X)
    X_test = X_test_unnorm - np.min(X)

    print("X_train max", np.max(X_train))
    print("X_train min", np.min(X_train))
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train, shape: ", y_train.shape)
    print("y_test, shape: ", y_test.shape)

    # coarsening the PPI graph to perform pooling in the model
    graphs, perm = coarsening.coarsen(A, levels=2, self_connections=False)
    L = [graph.laplacian(A, normalized=True) for A in graphs]

    X_train = coarsening.perm_data(X_train, perm)
    X_test = coarsening.perm_data(X_test, perm)

    n_train = X_train.shape[0]

    params = dict()
    params['dir_name']       = 'demo'
    params['num_epochs']     = 100
    params['batch_size']     = 109
    params['eval_frequency'] = 40

    # Building blocks.
    params['filter']         = 'chebyshev5'
    params['brelu']          = 'b1relu'
    params['pool']           = 'mpool1'

    # Number of classes.
    C = y.max() + 1
    assert C == np.unique(y).size

    # Architecture.
    params['F']              = [32, 32]  # Number of graph convolutional filters.
    params['K']              = [8, 8]  # Polynomial orders.
    params['p']              = [2, 2]    # Pooling sizes.
    params['M']              = [512, 128, C]  # Output dimensionality of fully connected layers.

    # Optimization.
    params['regularization'] = 1e-4
    params['dropout']        = 1
    params['learning_rate']  = 1e-3
    params['decay_rate']     = 0.95
    params['momentum']       = 0
    params['decay_steps']    = n_train / params['batch_size']

    model = models.cgcnn(L, **params)

    # !!!
    # TRAINING. Uncomment if you want to retrain the model.
    # start = time.time()
    # accuracy, loss, t_step, trained_losses = model.fit(X_train, y_train, X_test, y_test)
    # end = time.time()

    probas_ = model.get_probabilities(X_test)
    labels_by_network = np.argmax(probas_, axis=1)

    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    f1 = 100 * f1_score(y_test, labels_by_network, average='weighted')
    acc = accuracy_score(y_test, labels_by_network)
    print("\n\tTest AUC:", roc_auc) # np.argmax(y_preds, axis=2)[:, 0] fot categorical
    print("\tTest F1 weighted: ", f1)
    print("\tTest Accuraccy:", acc, "\n")

    # !!!
    # Creating hot-encoded labels for GLRP
    I = np.eye(C)
    tmp = I[labels_by_network]
    labels_hot_encoded = np.ones((model.batch_size, C))
    labels_hot_encoded[0:tmp.shape[0], 0:tmp.shape[1]] = tmp
    print("labels_hot_encoded.shape", labels_hot_encoded.shape)

    dir_to_save = "./results/"

    print("labels_by_network type", labels_by_network.dtype)
    print("y_test type", y_test.dtype)
    concordance = y_test == labels_by_network
    concordance = concordance.astype(int)
    out_labels_conc_df = pd.DataFrame(np.array([labels_by_network, concordance]).transpose(),
                                      columns=["Predicted", "Concordance"])
    concordance_df = patient_ind_test_df.join(out_labels_conc_df)
    concordance_df.to_csv(path_or_buf = dir_to_save + "predicted_concordance.csv", index=False)

    # !!!
    # CALCULATION OF RELEVANCES
    # CAN TAKE QUITE SOME TIME (UP to 10 MIN, Intel(R) Xeon(R) CPU E5-1620 v2 @ 3.70GHz, 32 GB RAM)
    glrp = glrp_scipy.GraphLayerwiseRelevancePropagation(model, X_test, labels_hot_encoded)
    rel = glrp.get_relevances()[-1][:X_test.shape[0], :]
    rel = coarsening.perm_data_back(rel, perm, X.shape[1])
    rel_df = pd.DataFrame(rel, columns=DP.feature_names)
    rel_df = pd.DataFrame(data={"Patient ID": patient_indexes_test}).join(rel_df)
    rel_df.to_csv(path_or_buf = dir_to_save + "relevances_rendered_class.csv", index=False)

