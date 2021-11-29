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
import math


def gen_codebook(codebook_type, m, d):
    if codebook_type == "Gaussian":
        mu = d*[0]
        cov = np.random.normal(0, 1, size=(d, d))
        cov = np.dot(cov, cov.transpose())
        cov_diag = cov.diagonal()
        cov = cov / np.sum(cov_diag)
        rv = multivariate_normal(mu, cov)
        return rv.rvs(m), cov  # codebook is m x d
    if codebook_type == "Grid":
        codewords_per_axis = int(np.ceil(m**(1/d)))
        # grid = list(multiset_permutations(np.linspace(-1, 1, codewords_per_axis), d))
        # repetitions = [d*[e] for e in np.linspace(-1, 1, codewords_per_axis) if e != 0]
        # complete_grid = np.array(grid+repetitions)
        grid = [list(p) for p in itertools.product(np.linspace(-1, 1, codewords_per_axis), repeat=d)]
        if codewords_per_axis % 2 == 1:
            grid.remove(d*[0.0])
        grid = np.array(grid)
        indexlist = np.argsort(np.linalg.norm(grid, axis=1))
        return grid[indexlist[:m]], None
    if codebook_type == "Circle":
        pi = math.pi
        return np.array([(math.cos(2*pi/m*x), math.sin(2*pi/m*x)) for x in range(m)]), None
    if codebook_type == "TwoCircles":
        pi = math.pi
        outer_words = int(2*m/3)
        inner_words = m - outer_words
        outer_circle = [(math.cos(2*pi/outer_words*x), math.sin(2*pi/outer_words*x)) for x in range(outer_words)]
        inner_circle = [(0.5*math.cos(2*pi/inner_words*x), 0.5*math.sin(2*pi/inner_words*x)) for x in range(inner_words)]
        return np.array(outer_circle+inner_circle), None
    if codebook_type == "GridInCircle":
        pi = math.pi
        outer_words = int(2*m/3)
        inner_words = m - outer_words
        outer_circle = [(math.cos(2*pi/outer_words*x), math.sin(2*pi/outer_words*x)) for x in range(outer_words)]
        codewords_per_axis = int(np.ceil(inner_words**(1/d)))
        inner = [list(p) for p in itertools.product(np.linspace(-0.3, 0.3, codewords_per_axis), repeat=d)]
        if codewords_per_axis % 2 == 1:
            inner.remove(d*[0.0])
        return np.array(outer_circle+inner), None


def gen_noise_dataset(noise_type, n, d, noise_energy, noise_cov=None, mix_dist=None):
    cov = None
    if noise_type == "Gaussian":
        mu = d*[0]
        if noise_cov is None:
            cov = np.random.normal(0, 1, size=(d, d))
            cov = np.dot(cov, cov.transpose())
            cov_diag = cov.diagonal()
            cov = (noise_energy/np.sum(cov_diag))*cov
        else:
            cov_diag = noise_cov.diagonal()
            cov = (noise_energy/np.sum(cov_diag))*noise_cov
        rv = multivariate_normal(mu, cov)
        return rv.rvs(n), cov, None  # noise samples are n x d
    if noise_type == "Mixture":
        mu = d * [0]
        if noise_cov is None:
            n_gaussians = np.random.randint(3, 10)
            mixture_dist = np.abs(np.random.normal(0, 1, size=n_gaussians))
            mixture_dist = mixture_dist / np.sum(mixture_dist)
            covs = np.random.normal(0, 1, size=(n_gaussians, d, d))
            for i, cov in enumerate(covs):
                cov = np.dot(cov, cov.transpose())
                cov_diag = cov.diagonal()
                covs[i] = (noise_energy / np.sum(cov_diag)) * cov
        else:
            n_gaussians = len(noise_cov)
            mixture_dist = mix_dist
            covs = noise_cov
            for i, cov in enumerate(covs):
                cov_diag = cov.diagonal()
                covs[i] = (noise_energy / np.sum(cov_diag)) * cov
        rvs = [multivariate_normal(mu, covs[i]) for i in range(n_gaussians)]
        mixture_idx = np.random.choice(len(mixture_dist), size=n, replace=True, p=mixture_dist)
        samples = np.array([rvs[idx].rvs(1) for idx in mixture_idx])
        return samples, covs, mixture_dist


def dataset_transform(codebook, noise_dataset, m, n, d):
    dataset = np.zeros((m, n, d))  # dataset is m x n x d
    for i in range(len(codebook)):
        dataset[i] = noise_dataset + codebook[i]
    return dataset


def plot_dataset(dataset, m, snr):
    fig = plt.figure()
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
    for i in range(m):
        ax.scatter(dataset[i, :, 0], dataset[i, :, 1], marker='x', s=10)
    ax.set_title("Labels")
    plt.grid()
    plt.savefig('True_Labels_'+str(snr).split(".")[0]+'_'+str(snr).split(".")[1])
    plt.close()


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


def decode_sample(a, codebook, S):
    return np.argmin([distance.mahalanobis(a, c, S) for c in codebook])


def decode(codebook, dataset, m, n, d, S):
    examples = dataset.reshape(m*n, d)
    classification = np.apply_along_axis(decode_sample, 1, examples, codebook, S)
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
    plt.close()


def gen_partition(d, deltas):
    basis = []
    additional_partition = []
    for delta in deltas:
        basis_pre = np.asarray(basis)
        rank_pre = matrix_rank(basis_pre)
        basis.append(delta)
        basis_post = np.asarray(basis)
        rank_post = matrix_rank(basis_post)
        if rank_post == rank_pre:
            additional_partition.append(basis.pop())
    P_arr = []
    for vector in basis:
        P_arr.append(np.array([vector]))
    P_arr.append(np.array(additional_partition))
    return P_arr


def make_run_dir(load, load_dir):
    if not os.path.exists("runs"):
        os.mkdir("runs")
    os.chdir("runs")
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H%M%S")
    if load:
        fin_string = dt_string + "_load_" + load_dir
    else:
        fin_string = dt_string + "_save"
    os.mkdir(fin_string)
    os.chdir(fin_string)


def plot_error_rate(train_errors, cov_train_errors, test_errors, cov_test_errors):
    for i in range(2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize='medium', width=3)
        plt.yscale('log')
        plt.grid()
        if i == 0:
            ax.plot(train_errors, linewidth=2, color='blue')
            ax.plot(cov_train_errors, color='black', linestyle='dashed', linewidth=2)
            plt.savefig('Train_Error_Probability')
        else:
            ax.plot(test_errors, linewidth=2, color='blue')
            ax.plot(cov_test_errors, color='black', linestyle='dashed', linewidth=2)
            plt.savefig('Test_Error_Probability')
        plt.close()


def plot_snr_error_rate(errors, cov_errors, snr_range, org_snr, codebook_energy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    snr_values = [codebook_energy/energy for energy in snr_range]
    snr_values = 10*np.log10(snr_values)
    print(snr_values)
    print(errors)
    print(cov_errors)
    ax.plot(snr_values, errors, color='blue', marker='s', linewidth=2)
    ax.plot(snr_values, cov_errors, color='black', linestyle='dashed', marker='s', linewidth=2)
    ax.tick_params(labelsize='medium', width=3)
    plt.axvline(x=10*np.log10(codebook_energy/org_snr))
    # plt.yscale('symlog', linthresh=10**-7)
    plt.yscale('log')
    # plt.ylim([-10**-7, 1])
    plt.grid()
    plt.savefig('Error_Probability_SNR', dpi=300)
    plt.close()
    f = open('SNR_errors.npy', 'wb')
    np.save(f, errors)
    f.close()
    f = open('SNR_cov_errors.npy', 'wb')
    np.save(f, cov_errors)
    f.close()
    f = open('SNR_range.npy', 'wb')
    np.save(f, snr_range)
    f.close()


