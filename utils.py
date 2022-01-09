from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy import linalg as LA
from datetime import datetime
import os
from sympy.utilities.iterables import multiset_permutations
import itertools
import math
import cvxpy as cp


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
    if noise_type in ["Gaussian", "WhiteGaussian"]:
        mu = d*[0]
        if noise_cov is None:
            if noise_type == "WhiteGaussian":
                cov = (noise_energy/d)*np.eye(d)
            else:
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


def gen_transformation(d_x, d_y, trans_type, max_eigenvalue, min_eigenvalue):
    if trans_type == "Linear_Invertible" and d_x != d_y:
        print("Asking for invertible non-square matrix")
        exit()

    if trans_type in ["Linear", "Linear_Invertible", "Rotate"]:
        trans = np.random.rand(d_y, d_x)
        if trans_type == "Linear_Invertible":
            trans = trans.T @ trans
            trans = trans + trans.T
        u, s, vh = np.linalg.svd(trans, full_matrices=True)
        curr_max_eigen = np.max(s)
        trans = (max_eigenvalue / curr_max_eigen) * trans
        u, s, vh = np.linalg.svd(trans, full_matrices=True)
        s[s < min_eigenvalue] = min_eigenvalue
        trans = np.dot(u[:, :d_x] * s, vh)
        if trans_type == "Rotate" and d_x == 2 and d_y == 2:
            sigma1 = np.random.uniform(min_eigenvalue, max_eigenvalue)
            trans = [[0, -sigma1], [sigma1, 0]]

        def f(x):
            return trans @ x

        f_kernel = trans

        def f_inv(y):
            return LA.pinv(trans) @ y

    if trans_type in ["Quadratic"]:
        _, _, a = gen_transformation(d_x, d_y, "Rotate", max_eigenvalue, min_eigenvalue)
        _, _, b = gen_transformation(d_x, d_y, "Linear", 0.2*max_eigenvalue, 0.2*min_eigenvalue)

        def f(x):
            return a @ x + b @ x**2

        f_kernel = (a, b)

        def f_inv(y, cb):
            return np.argmin([distance.euclidean(a@c+b@c**2, y) for c in cb])

    if trans_type == "Identity":

        def f(x):
            return np.pad(x, ((0, d_y-d_x), (0, 0)), 'constant')

        f_kernel = np.pad(np.eye(d_x), ((0, d_y-d_x), (0, 0)), 'constant')

        def f_inv(y):
            if d_x == d_y:
                return y
            return y[:-(d_y-d_x)][:]

    return f, f_inv, f_kernel


def rebuild_trans_from_kernel(f_kernel, trans_type):
    if trans_type in ["Linear", "identity", "Rotate"]:

        def f(x):
            return f_kernel @ x

        def f_inv(y, cb):
            return np.argmin([distance.euclidean(f_kernel@c, y) for c in cb])

    if trans_type in ["Quadratic"]:

        def f(x):
            return f_kernel[0] @ x + f_kernel[1] @ x**2

        def f_inv(y, cb):
            return np.argmin([distance.euclidean(f_kernel[0]@c+f_kernel[1]@c**2, y) for c in cb])

    return f, f_inv


def dataset_transform(codebook, noise_dataset, m, n, d):
    dataset = np.zeros((m, n, d))  # dataset is m x n x d
    for i in range(len(codebook)):
        dataset[i] = noise_dataset + codebook[i]
    return dataset


def dataset_transform_LTNN(codebook, noise_dataset, m, n, trans):
    transformed_codewords = trans(codebook.T)  # d_x x m
    dup_trans_codewords = np.repeat(transformed_codewords.T, int(n/m), axis=0)  # n x d_y
    return dup_trans_codewords + noise_dataset


def plot_dataset(dataset, m, snr, codebook, inv_trans):
    for j in range(1):
        fig = plt.figure()
        cm = plt.get_cmap('gist_rainbow')
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
        for i in range(m):
            ax.scatter(codebook[i][0], codebook[i][1], marker='o', s=50)
        ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
        if j == 0:
            for i in range(m):
                ax.scatter(dataset[i*int(len(dataset)/m):(i+1)*int(len(dataset)/m)-1, 0],
                           dataset[i*int(len(dataset)/m):(i+1)*int(len(dataset)/m)-1, 1], marker='x', s=10)
        else:
            for i in range(m):
                inv_tans_dataset = np.array(dataset[i*int(len(dataset)/m):(i+1)*int(len(dataset)/m)])
                inv_tans_dataset = inv_trans(inv_tans_dataset.T)
                inv_tans_dataset = inv_tans_dataset.T
                ax.scatter(inv_tans_dataset[:, 0], inv_tans_dataset[:, 1], marker='x', s=10)
        plt.grid()
        if j == 0:
            plt.savefig('Codebook_and_samples_'+str(snr).split(".")[0]+'_'+str(snr).split(".")[1])
        else:
            plt.savefig('Codebook_and_inv_transformed_samples_'+str(snr).split(".")[0]+'_'+str(snr).split(".")[1])
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


def decode_sample_LTNN(a, cb, h):
    return np.argmin([LA.norm(a - h @ c) for c in cb])


def decode_LTNN(codebook, dataset, m, n, d, H):
    classification = np.apply_along_axis(decode_sample_LTNN, 1, dataset, codebook, H)
    return classification


def trans_decode(codebook, dataset, inv_trans):
    classification = np.apply_along_axis(inv_trans, 1, dataset, codebook)
    return classification


def plot_decoding(dataset, classification, m, n, d, t):
    fig = plt.figure()
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / m) for i in range(m)])
    for i in range(m):
        ax.scatter(dataset[np.where(classification == i), 0], dataset[np.where(classification == i), 1],
                   marker='x', s=10)
    ax.set_title("Classification")
    plt.grid()
    plt.savefig('Iteration_'+str(t).zfill(6))
    plt.close()


def gen_partition(deltas):
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
    os.chdir("runs/LTNN")
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H%M%S")
    if load:
        fin_string = dt_string + "_load_" + load_dir
    else:
        fin_string = dt_string + "_save"
    os.mkdir(fin_string)
    os.chdir(fin_string)


def plot_error_rate(train_errors, cov_train_errors, test_errors, cov_test_errors, lambda_scale=None, iter_gap=1):
    for i in range(2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize='medium', width=3)
        plt.yscale('log')
        plt.grid()
        iter_axis = [iter_gap*j for j in range(len(train_errors))]
        if i == 0:
            ax.plot(iter_axis, train_errors, linewidth=2, color='blue')
            ax.plot(iter_axis, cov_train_errors, color='black', linestyle='dashed', linewidth=2)
            plt.savefig('Train_Error_Probability_'+str(lambda_scale).replace(".", "_"))
        else:
            ax.plot(iter_axis, test_errors, linewidth=2, color='blue')
            ax.plot(iter_axis, cov_test_errors, color='black', linestyle='dashed', linewidth=2)
            plt.savefig('Test_Error_Probability_'+str(lambda_scale).replace(".", "_"))
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
    plt.savefig('Error_Probability_SNR')
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


def plot_indicator(lto, lambda_range):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.logspace(lambda_range[0], lambda_range[1], 50), lto, color='blue', marker='s', linewidth=2)
    plt.grid()
    plt.savefig('plot_indicator')
    plt.close()


def plot_norms(h_array, s_array, scale_lambda):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([LA.norm(h) for h in h_array], color='blue', marker='s', linewidth=2)
    ax.plot([LA.norm(s) for s in s_array], color='red', marker='s', linewidth=2)
    plt.grid()
    plt.savefig('h_norm_blue_s_norm_red_'+str(scale_lambda).replace(".", "_"))
    plt.close()


def projection(h1, s1):
    h2 = cp.Variable(h1.shape)
    s2 = cp.Variable(s1.shape, PSD=True)
    obj = cp.Minimize(cp.square(cp.norm(s2 - s1, 'fro'))+cp.square(cp.norm(h2 - h1, 'fro')))
    LMI1 = cp.bmat([
        [np.eye(s2.shape[0]), h2],
        [h2.T, s2]
    ])
    constraints = [LMI1 >> 0]
    # constraints = [s2-h2.T@h2 >> 0]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)

    return h2.value, s2.value
