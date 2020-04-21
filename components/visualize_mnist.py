import numpy as np
from matplotlib import pyplot as plt


def get_heatmap(digit):
    """Computes the heatmap of the out of the digit."""
    digit = np.abs(np.reshape(digit, newshape=(28, 28)))
    return (digit - np.min(digit))/(np.max(digit) - np.min(digit))


def plot_numbers(data_to_test, rel, labels_data_to_test, labels_by_network, results_dir):
    """Plots the numbers and the relevance heatmap of it. """
    w = 70
    h = 70
    fig = plt.figure(figsize=(15, 15))
    columns = 2
    rows = data_to_test.shape[0] # number of digits
    ax = []
    ax_ind = 0
    for i in range(0, int(columns*rows/2), 1):
        # create subplot and append to ax
        ax_ind +=1
        ax.append(fig.add_subplot(rows, columns, ax_ind))
        ax[-1].set_title("real:"+str(labels_data_to_test[i]))  # set title
        plt.imshow(get_heatmap(data_to_test[i, :]), cmap='Reds')

        ax_ind += 1
        ax.append(fig.add_subplot(rows, columns, ax_ind))
        ax[-1].set_title("predicted label:" + str(labels_by_network[i]))  # set title
        plt.imshow(get_heatmap(rel[i, :]), cmap='Reds'),# interpolation='bilinear')

    plt.savefig("{0}results_grid_not_weighted.png".format(results_dir))  # finally, render the plot
