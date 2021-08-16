from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from datetime import datetime
import os
from sympy.utilities.iterables import multiset_permutations
import itertools


def gen_codebook(codebook_type, m, d):
    if codebook_type == "Gaussian":
        mu = d*[0]
        cov_diag = np.random.normal(0, 1, size=d)
        scaled_cov_diag = cov_diag/np.sum(cov_diag)
        cov = np.diag(scaled_cov_diag)
        rv = multivariate_normal(mu, cov)
        return rv.rvs(m), cov  # codebook is m x d
    if codebook_type == "Grid":
        codewords_per_axis = int(np.ceil(m**(1/d)))
        # grid = list(multiset_permutations(np.linspace(-1, 1, codewords_per_axis), d))
        # repetitions = [d*[e] for e in np.linspace(-1, 1, codewords_per_axis) if e != 0]
        # complete_grid = np.array(grid+repetitions)
        grid = [list(p) for p in itertools.product(np.linspace(-1, 1, codewords_per_axis), repeat=d)]
        grid.remove(d*[0.0])
        grid = np.array(grid)
        indexlist = np.argsort(np.linalg.norm(grid, axis=1))
        return grid[indexlist[:m]], None


def gen_noise_dataset(noise_type, n, d, noise_energy):
    cov = None
    if noise_type == "Gaussian":
        mu = d*[0]
        cov = np.random.normal(0, 1, size=(d, d))
        cov = np.dot(cov, cov.transpose())
        cov_diag = cov.diagonal()
        # scaled_cov_diag = noise_energy*cov_diag/np.sum(cov_diag)
        # np.fill_diagonal(cov, scaled_cov_diag)
        cov = (noise_energy/np.sum(cov_diag))*cov
        rv = multivariate_normal(mu, cov)
        return rv.rvs(n), cov  # noise samples are n x d
    if noise_type == "Mixture":
        n_gaussians = np.random.randint(10)
        mixture_dist = np.abs(np.random.normal(0, 1, size=n_gaussians))
        mixture_dist = mixture_dist/np.sum(mixture_dist)
        mu = d * [0]
        covs = np.random.normal(0, 1, size=(n_gaussians, d, d))
        for i, cov in enumerate(covs):
            cov = np.dot(cov, cov.transpose())
            cov_diag = cov.diagonal()
            # scaled_cov_diag = noise_energy * cov_diag / np.sum(cov_diag)
            # np.fill_diagonal(cov, scaled_cov_diag)
            covs[i] = (noise_energy / np.sum(cov_diag)) * cov
        rvs = [multivariate_normal(mu, covs[i]) for i in range(n_gaussians)]
        mixture_idx = np.random.choice(len(mixture_dist), size=n, replace=True, p=mixture_dist)
        samples = np.array([rvs[idx].rvs(1) for idx in mixture_idx])
        return samples, covs


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
        ax.scatter(dataset[i, :, 0], dataset[i, :, 1], marker='x', s=10)
    ax.set_title("Labels")
    plt.grid()
    plt.savefig('True_Labels')


def delta_array(L, d, m, codebook):
    deltas = np.zeros((L, d))
    for i in range(m):
        for j in range(i+1, m):
            deltas[double_to_single_index(i, j, m)] = codebook[i] - codebook[j]
    return deltas


def get_near_psd(s):
    eigval, eigvec = np.linalg.eig(s)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


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


def plot_decoding(dataset, classification, m, n, d, t):
    fig = plt.figure()
    examples = dataset.reshape(m * n, d)
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
    for i in range(m):
        ax.scatter(examples[np.where(classification == i), 0], examples[np.where(classification == i), 1],
                   marker='x', s=10)
    ax.set_title("Classification")
    plt.grid()
    plt.savefig('Iteration_'+str(t).zfill(6))


def gen_partition(d, deltas):
    P = []
    for delta in deltas:
        succ = False
        for p_i in P:
            p_i_pre = np.asarray(p_i)
            rank_pre = matrix_rank(p_i_pre)
            p_i.append(delta)
            p_i_post = np.asarray(p_i)
            rank_post = matrix_rank(p_i_post)
            if rank_post == rank_pre:
                succ = True
                break
        if not succ:
            if len(P) < d:
                new_p = [delta]
                P.append(new_p)
            else:
                P[np.random.randint(d)].append(delta)
    P_arr = []
    for p_i in P:
        P_arr.append(np.array(p_i))
    return P_arr


def make_run_dir():
    if not os.path.exists("runs"):
        os.mkdir("runs")
    os.chdir("runs")
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H%M%S")
    os.mkdir(dt_string)
    os.chdir(dt_string)


def plot_error_rate(errors):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(errors)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error Probability")
    plt.grid()
    plt.savefig('ErrorProbability')

