from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv


def gen_codebook(m, dim):  # TODO: control the norm of the codewords, specified by x_norm_max
    mu = [0, 0]  # TODO: not hard coded mu and cov
    cov = [[1, 0.5], [0.5, 1]]
    rv = multivariate_normal(mu, cov)
    return rv.rvs(m)  # codebook is m x d


def gen_noise_dataset(noise_type, n, dim):
    if noise_type == "Gaussian":
        mu = [0, 0]  # TODO: not hard coded mu and cov
        cov = [[0.1, 0.05], [0.05, 0.1]]
        rv = multivariate_normal(mu, cov)
        return rv.rvs(n)  # noise samples are n x d


def dataset_transform(codebook, noise_dataset, m, n, d):
    dataset = np.zeros((m, n, d))  # dataset is m x n x d
    for i in range(len(codebook)):
        dataset[i] = noise_dataset + codebook[i]
    return dataset


def plot_dataset(dataset, m, fig):
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
    for i in range(m):
        ax.scatter(dataset[i, :, 0], dataset[i, :, 1])
    ax.set_title("Labels")
    plt.grid()
    plt.savefig('True_Labels')


def delta_array(L, d, m, codebook):
    deltas = np.zeros((L, d))
    for i in range(m):
        for j in range(i+1, m):
            deltas[double_to_single_index(i, j, m)] = codebook[i] - codebook[j]
    return deltas


def double_to_single_index(i, j, m):
    l = 0
    for k in range(i, 0, -1):
        l += m-k
    l += j - i - 1
    return l


def single_to_double_index(l, m):
    reminder = l
    i = 0
    while reminder >= m-i-1:
        reminder -= m-i-1
        i += 1
    j = reminder - i
    return i, j


def decode(codebook, dataset, m, n, d, S):
    examples = dataset.reshape(m*n, d)
    classification = np.zeros(m*n)
    iv = inv(S)
    for i, e in enumerate(examples):
        c = np.argmin([distance.mahalanobis(e, c, iv) for c in codebook])
        classification[i] = c
    return classification


def plot_decoding(dataset, classification, m, n, d, t, fig):
    examples = dataset.reshape(m * n, d)
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
    for i in range(m):
        ax.scatter(examples[np.where(classification == i), 0], examples[np.where(classification == i), 1])
    ax.set_title("Classification")
    plt.grid()
    plt.savefig('imgs/Iteration_'+str(t).zfill(6))