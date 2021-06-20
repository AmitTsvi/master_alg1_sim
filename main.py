import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from numpy.linalg import inv
from numpy import linalg as LA
import utils


def init_precision_matrix(d):
    a = np.random.rand(d, d)
    return np.dot(a, a.T)


def get_near_psd(s):
    eigval, eigvec = np.linalg.eig(s)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def plot_pegasos(s_array, codebook, dataset, m, n, d):
    for t in range(len(s_array)):
        classification = utils.decode(codebook, dataset, m, n, d, s_array[t])
        if t % 10 == 0:
            utils.plot_decoding(dataset, classification, m, n, d, t, fig)


def subgradient_alg(steps, m, n, deltas, eta, d, codebook, dataset):
    s_array = []
    s = np.zeros((d, d))
    for t in range(1, steps+1):
        p_t = np.random.randint(m-1)
        q_t = np.random.randint(p_t+1, m)
        x_p_t = np.expand_dims(codebook[p_t], axis=1)
        x_q_t = np.expand_dims(codebook[q_t], axis=1)
        z_t = np.random.randint(n)
        which_word = np.random.randint(2)
        y_t = np.expand_dims(dataset[p_t, z_t] if which_word == 0 else dataset[q_t, z_t], axis=1)
        delta_p_q_t = np.expand_dims(deltas[utils.double_to_single_index(p_t, q_t, m)], axis=1)
        delta_p_q_star = np.expand_dims(deltas[np.argmax(LA.norm(np.dot(s, deltas.T), axis=0)**2)], axis=1)
        a_t = ((-1)**which_word)*(y_t-0.5*(x_p_t+x_q_t))
        grad_t = eta*(s@delta_p_q_star@delta_p_q_star.T+delta_p_q_star@delta_p_q_star.T@s)
        if a_t.T@s@delta_p_q_t < 1:
            grad_t -= 0.5*(delta_p_q_t@a_t.T+a_t@delta_p_q_t.T)
        s -= (1/(eta*t))*grad_t
        s = get_near_psd(s)
        s_array.append(s)
    return s_array


def main():
    d = 2  # dimension
    m = 8  # number of codewords
    x_norm_max = 10
    n = 100  # number of noise samples
    L = int(m*(m-1)/2)  # number of codewords pairs with i<j
    noise_type = "Gaussian"
    codebook = utils.gen_codebook(m, d)
    noise_dataset = utils.gen_noise_dataset(noise_type, n, d)
    dataset = utils.dataset_transform(codebook, noise_dataset, m, n, d)
    utils.plot_dataset(dataset, m, fig)
    s = init_precision_matrix(d)
    deltas = utils.delta_array(L, d, m, codebook)
    s_array = subgradient_alg(100, m, n, deltas, 0.1, d, codebook, dataset)
    plot_pegasos(s_array, codebook, dataset, m, n, d)


if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    fig = plt.figure()
    main()




