from scipy.stats import multivariate_normal
from scipy.stats import multivariate_t
from scipy.stats import norm
from scipy.stats import gennorm
from scipy.stats import uniform
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
import cvxpy as cp


def gen_codebook(basic_dict):
    if basic_dict['codebook_type'] == "Gaussian":
        mu = basic_dict['d_x']*[0]
        cov = np.random.normal(0, 1, size=(basic_dict['d_x'], basic_dict['d_x']))
        cov = np.dot(cov, cov.transpose())
        cov_diag = cov.diagonal()
        cov = cov / np.sum(cov_diag)
        rv = multivariate_normal(mu, cov)
        return rv.rvs(basic_dict['m']), cov  # codebook is m x d
    if basic_dict['codebook_type'] == "Grid":
        codewords_per_axis = int(np.ceil(basic_dict['m']**(1/basic_dict['d_x'])))
        grid = [list(p) for p in itertools.product(np.linspace(-1, 1, codewords_per_axis), repeat=basic_dict['d_x'])]
        if codewords_per_axis % 2 == 1:
            grid.remove(basic_dict['d_x']*[0.0])
        grid = np.array(grid)
        indexlist = np.argsort(np.linalg.norm(grid, axis=1))
        return grid[indexlist[:basic_dict['m']]], None
    if basic_dict['codebook_type'] == "Circle":
        pi = math.pi
        return np.array([(math.cos(2*pi/basic_dict['m']*x), math.sin(2*pi/basic_dict['m']*x)) for x in range(basic_dict['m'])]), None
    if basic_dict['codebook_type'] == "TwoCircles":
        pi = math.pi
        outer_words = basic_dict['m'] // 2
        inner_words = basic_dict['m'] - outer_words
        outer_radius = 0.5+0.5*(3**0.5)
        inner_radius = 2**-0.5
        outer_circle = [(outer_radius*math.cos(2*pi/outer_words*x), outer_radius*math.sin(2*pi/outer_words*x)) for x in range(outer_words)]
        inner_circle = [(inner_radius*math.cos(2*pi/inner_words*x+pi/4), inner_radius*math.sin(2*pi/inner_words*x+pi/4)) for x in range(inner_words)]
        return np.array(outer_circle+inner_circle), None
    if basic_dict['codebook_type'] == "GridInCircle":
        pi = math.pi
        outer_words = int(2*basic_dict['m']/3)
        inner_words = basic_dict['m'] - outer_words
        outer_circle = [(math.cos(2*pi/outer_words*x), math.sin(2*pi/outer_words*x)) for x in range(outer_words)]
        codewords_per_axis = int(np.ceil(inner_words**(1/basic_dict['d_x'])))
        inner = [list(p) for p in itertools.product(np.linspace(-0.3, 0.3, codewords_per_axis), repeat=basic_dict['d_x'])]
        if codewords_per_axis % 2 == 1:
            inner.remove(basic_dict['d_x']*[0.0])
        return np.array(outer_circle+inner), None
    if basic_dict['codebook_type'] == "Custom":
        return np.array([[-1, -1], [1, 1]]), None


def gen_multimodal_means(basic_dict, n_gaussians, mix_dist):
    means = []
    reminder = n_gaussians % 2
    for i in range(int(basic_dict['d_y'])):
        select = np.random.randint(n_gaussians-reminder, size=n_gaussians//2)
        line = [0.5*basic_dict['noise_energy']*(-1)**(i in select) for i in range(n_gaussians-reminder)]
        if reminder:
            line.append(0)
        means.append(np.divide(line, mix_dist))
    return np.transpose(np.array(means))


def gen_noise_dataset(basic_dict, n, noise_cov=None, mix_means=None, mix_dist=None, noise_energy=None, df=None):
    if noise_energy is None:
        noise_energy = basic_dict['noise_energy']
    if basic_dict['noise_type'] in ["Gaussian", "WhiteGaussian", "student_t"]:
        mu = basic_dict['d_y']*[0]
        if noise_cov is None:
            if basic_dict['noise_type'] == "WhiteGaussian":
                cov = (noise_energy/basic_dict['d_y'])*np.eye(basic_dict['d_y'])
            else:
                cov = np.random.normal(0, 1, size=(basic_dict['d_y'], basic_dict['d_y']))
                cov = np.dot(cov, cov.transpose())
                cov_diag = cov.diagonal()
                cov = (noise_energy/np.sum(cov_diag))*cov
        else:
            cov_diag = noise_cov.diagonal()
            cov = (noise_energy/np.sum(cov_diag))*noise_cov
        if basic_dict['noise_type'] in ["Gaussian", "WhiteGaussian"]:
            rv = multivariate_normal(mu, cov)
            d_freedom = None
        else:
            if df is None:
                d_freedom = np.random.randint(low=1, high=11)
            else:
                d_freedom = df
            rv = multivariate_t(shape=cov, df=d_freedom)
        return rv.rvs(n), cov, None, None, d_freedom  # noise samples are n x d
    if basic_dict['noise_type'] == "Mixture":
        if noise_cov is None:
            n_gaussians = 2*np.random.randint(1, 3)
            mixture_dist = np.abs(np.random.normal(0, 1, size=n_gaussians))
            mixture_dist = mixture_dist / np.sum(mixture_dist)
            covs = np.random.normal(0, 1, size=(n_gaussians, basic_dict['d_y'], basic_dict['d_y']))
            means = gen_multimodal_means(basic_dict, n_gaussians, mixture_dist)
            for i, cov in enumerate(covs):
                cov = np.dot(cov, cov.transpose())
                cov_diag = cov.diagonal()
                covs[i] = (noise_energy / np.sum(cov_diag)) * cov
        else:
            n_gaussians = len(noise_cov)
            mixture_dist = mix_dist
            covs = noise_cov
            means = mix_means
            for i, cov in enumerate(covs):
                cov_diag = cov.diagonal()
                covs[i] = (noise_energy / np.sum(cov_diag)) * cov
        rvs = [multivariate_normal(means[i], covs[i]) for i in range(n_gaussians)]
        mixture_idx = np.random.choice(len(mixture_dist), size=n, replace=True, p=mixture_dist)
        samples = np.array([rvs[idx].rvs(1) for idx in mixture_idx])
        return samples, covs, means, mixture_dist, None
    if basic_dict['noise_type'] == "Custom":
        rv1 = norm(0, np.sqrt(0.5*noise_energy))
        beta = 0.1
        #rv2 = gennorm(beta=beta, scale=np.sqrt(0.5*noise_energy/gennorm.var(beta=beta)))
        rv2 = uniform(loc=-3*0.5*noise_energy, scale=6*0.5*noise_energy)
        samples = np.column_stack((rv2.rvs(n), rv1.rvs(n)))
        return samples, (1/(0.5*noise_energy))*np.eye(2), np.array([0, 0]), None, None


def gen_transformation(d_x, d_y, trans_type, max_eigenvalue, min_eigenvalue):
    if trans_type == "Linear_Invertible" and d_x != d_y:
        print("Asking for invertible non-square matrix")
        exit()

    if trans_type in ["Linear", "Linear_Invertible", "Rotate"]:
        trans = np.random.rand(d_y, d_x)
        if trans_type == "Linear_Invertible":
            trans = trans.T @ trans
            trans = trans + trans.T
        u, s, vh = np.linalg.svd(trans, full_matrices=False)
        curr_max_eigen = np.max(s)
        trans = (max_eigenvalue / curr_max_eigen) * trans
        u, s, vh = np.linalg.svd(trans, full_matrices=False)
        s[s < min_eigenvalue] = min_eigenvalue
        trans = np.dot(u * s, vh)
        if trans_type == "Rotate" and d_x == 2 and d_y == 2:
            sigma1 = np.random.uniform(min_eigenvalue, max_eigenvalue)
            trans = [[0, -sigma1], [sigma1, 0]]

        def f(x):
            return trans @ x

        f_kernel = trans

    if trans_type in ["Quadratic"]:
        _, a = gen_transformation(d_x, d_y, "Rotate", max_eigenvalue, min_eigenvalue)
        _, b = gen_transformation(d_x**2, d_y, "Linear", 0.2*max_eigenvalue, 0.2*min_eigenvalue)

        def f(x):
            x_1 = np.expand_dims(x, 1)
            x_2 = np.expand_dims((x_1@x_1.T).flatten(), 1)
            res = a @ x_1 + b @ x_2
            return np.squeeze(res)

        f_kernel = (a, b)

    if trans_type == "Identity":

        def f(x):
            return np.pad(x, ((0, d_y-d_x), (0, 0)), 'constant')

        f_kernel = np.pad(np.eye(d_x), ((0, d_y-d_x), (0, 0)), 'constant')

    return f, f_kernel


def rebuild_trans_from_kernel(f_kernel, trans_type):
    if trans_type in ["Linear", "identity", "Rotate"]:
        def f(x):
            return f_kernel @ x

    if trans_type in ["Quadratic"]:
        def f(x):
            x_1 = np.expand_dims(x, 1)
            x_2 = np.expand_dims((x_1@x_1.T).flatten(), 1)
            res = f_kernel[0] @ x_1 + f_kernel[1] @ x_2
            return np.squeeze(res)

    return f


def plot_dataset(dataset, snr, codebook, basic_dict):
    fig = plt.figure()
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / basic_dict['m']) for i in range(basic_dict['m'])])
    for i in range(basic_dict['m']):
        ax.scatter(codebook[i][0], codebook[i][1], marker='o', s=50)
    ax.set_prop_cycle(color=[cm(1. * i / basic_dict['m']) for i in range(basic_dict['m'])])
    for i in range(basic_dict['m']):
        if basic_dict['model'] == "LTNN":
            ax.scatter(dataset[i*int(len(dataset)/basic_dict['m']):(i+1)*int(len(dataset)/basic_dict['m'])-1, 0],
                       dataset[i*int(len(dataset)/basic_dict['m']):(i+1)*int(len(dataset)/basic_dict['m'])-1, 1], marker='x', s=10)
        if basic_dict['model'] == "MNN":
            ax.scatter(dataset[i, :, 0], dataset[i, :, 1], marker='x', s=10)
    plt.grid()
    plt.title('Codebook and Output Samples')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig('Codebook_and_samples_'+str(snr).split(".")[0]+'_'+str(snr).split(".")[1])
    plt.close()


def delta_array(codebook, basic_dict):
    L = int(basic_dict['m'] * (basic_dict['m'] - 1) / 2)
    deltas = np.zeros((L, basic_dict['d_x']))
    for i in range(basic_dict['m']):
        for j in range(i+1, basic_dict['m']):
            deltas[double_to_single_index(i, j, basic_dict['m'])] = codebook[i] - codebook[j]
    return deltas


def get_near_psd(s):
    eigval, eigvec = np.linalg.eig(s)
    eigval[eigval < 0] = 0
    return eigvec@np.diag(eigval)@inv(eigvec)


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


def plot_decoding(dataset, classification, basic_dict, t, title=None):
    if not title:
        title = 'Iteration_'+str(t).zfill(6)
    fig = plt.figure()
    cm = plt.get_cmap('gist_rainbow')
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / basic_dict['m']) for i in range(basic_dict['m'])])
    for i in range(basic_dict['m']):
        if basic_dict['model'] == "LTNN":
            ax.scatter(dataset[np.where(classification == i), 0], dataset[np.where(classification == i), 1],
                       marker='x', s=10)
        if basic_dict['model'] == "MNN":
            dataset = dataset.reshape(-1, basic_dict['d_x'])
            ax.scatter(dataset[classification == i, 0], dataset[classification == i, 1], marker='x', s=10)
    ax.set_title("Classification")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid()
    plt.savefig(title+".png")
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
    if additional_partition:
        P_arr.append(np.array(additional_partition))
    return P_arr


def make_run_dir(load, load_dir, basic_dict):
    if not os.path.exists("runs"):
        os.mkdir("runs")
    os.chdir("runs/"+basic_dict['model'])
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H%M%S")
    if load:
        fin_string = dt_string + "_load_" + load_dir
    else:
        fin_string = dt_string + "_save"
    os.mkdir(fin_string)
    os.chdir(fin_string)


def plot_error_rate(train_errors, train_rule_error, train_naive_error, test_errors, test_rule_error, test_naive_error,
                    obj_vals, lambda_scale=None, iter_gap=1):
    for i in range(2):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize='medium', width=3)
        plt.yscale('log')
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Error Probability')
        iter_axis = [iter_gap*j for j in range(len(train_errors))]
        if i == 0:
            ax.plot(iter_axis, train_errors, linewidth=2, color='blue')
            ax.plot(iter_axis, len(iter_axis)*[train_rule_error], color='black', linestyle='dashed')
            ax.plot(iter_axis, len(iter_axis)*[train_naive_error], color='red', linestyle='dashed')
            ax2 = ax.twinx()
            ax2.plot(obj_vals[::iter_gap], color='orange')
            plt.yscale('log')
            plt.title('Train Error')
            plt.savefig('Train_Error_Probability_'+str(lambda_scale).replace(".", "_"))
        else:
            ax.plot(iter_axis, test_errors, linewidth=2, color='blue')
            ax.plot(iter_axis, len(iter_axis)*[test_rule_error], color='black', linestyle='dashed')
            ax.plot(iter_axis, len(iter_axis)*[test_naive_error], color='red', linestyle='dashed')
            plt.title('Test Error')
            plt.savefig('Test_Error_Probability_'+str(lambda_scale).replace(".", "_"))
        plt.close()


def plot_snr_error_rate(errors, labels, basic_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(basic_dict['snr_range'])
    print(errors)
    colors = ['blue', 'black', 'green', 'red']
    for index, error in enumerate(errors):
        ax.plot(basic_dict['snr_range'], error, color=colors[index], marker='s', linewidth=2, label=labels[index])
    ax.tick_params(labelsize='medium', width=3)
    plt.axvline(x=10*np.log10(basic_dict['code_energy']/basic_dict['noise_energy']))
    plt.legend()
    plt.yscale('log')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Error Probability')
    plt.grid()
    plt.savefig('Error_Probability_SNR')
    plt.close()
    f = open('all_errors.npy', 'wb')
    np.save(f, errors)
    f.close()


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
