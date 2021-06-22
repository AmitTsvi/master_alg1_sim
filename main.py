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
            utils.plot_decoding(dataset, classification, m, n, d, t)


def subgradient_alg(steps, m, n, deltas, etas, d, codebook, dataset, scale_lambda, partition):
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
        a_t = ((-1)**which_word)*(y_t-0.5*(x_p_t+x_q_t))
        v_t = 0
        if a_t.T@s@delta_p_q_t < 1:
            v_t = 0.5*(delta_p_q_t@a_t.T+a_t@delta_p_q_t.T)
        grad_t = 0
        for i, p_i in enumerate(partition):
            delta_p_q_star = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
            grad_t += etas[i] * (s @ delta_p_q_star @ delta_p_q_star.T + delta_p_q_star @ delta_p_q_star.T @ s)
        grad_t = scale_lambda*grad_t - v_t
        s -= (1/(scale_lambda*t))*grad_t
        s = get_near_psd(s)
        s_array.append(s)
    return s_array


def log_run_info(d, m, n, codebook, iterations, scale_lambda, etas, codebook_type, codeword_energy, noise_type,
                 noise_energy, code_cov, noise_cov):
    file1 = open("log.txt", "w")
    file1.write("Dimension "+str(d)+"\n")
    file1.write("Number of codewords " + str(m)+"\n")
    file1.write("Codebook type:" + codebook_type + "\n")
    file1.write("codewords energy " + str(codeword_energy)+"\n")
    if code_cov is not None:
        file1.write("Random codebook covariance:" + "\n")
        file1.write(str(code_cov) + "\n")
    file1.write("Codebook:"+"\n")
    file1.write(str(codebook)+"\n")
    file1.write("Noise type:" + noise_type + "\n")
    if code_cov is not None:
        file1.write("Random noise covariance:" + "\n")
        file1.write(str(noise_cov) + "\n")
    file1.write("codewords energy " + str(noise_energy) + "\n")
    file1.write("Number of iterations " + str(iterations)+"\n")
    file1.write("Lambda " + str(scale_lambda)+"\n")
    file1.write("Etas " + str(etas)+"\n")
    file1.write("Seed " + str(seed) + "\n")
    file1.close()


def main():
    utils.make_run_dir()
    d = 2  # dimension
    m = 32  # number of codewords
    x_norm_max = 10
    n = 100  # number of noise samples
    L = int(m*(m-1)/2)  # number of codewords pairs with i<j
    iterations = 100
    scale_lambda = 0.1
    etas = d*[1/d]
    codebook_type = "Grid"
    codeword_energy = 1
    noise_type = "Gaussian"
    noise_energy = 0.005
    codebook, code_cov = utils.gen_codebook(codebook_type, m, d)
    noise_dataset, noise_cov = utils.gen_noise_dataset(noise_type, n, d, noise_energy)
    dataset = utils.dataset_transform(codebook, noise_dataset, m, n, d)
    utils.plot_dataset(dataset, m, fig)
    s = init_precision_matrix(d)
    deltas = utils.delta_array(L, d, m, codebook)
    partition = utils.gen_partition(d, deltas)
    s_array = subgradient_alg(iterations, m, n, deltas, etas, d, codebook, dataset, scale_lambda, partition)
    plot_pegasos(s_array, codebook, dataset, m, n, d)
    log_run_info(d, m, n, codebook, iterations, scale_lambda, etas, codebook_type, codeword_energy, noise_type,
                 noise_energy, code_cov, noise_cov)


if __name__ == '__main__':
    seed = 8
    np.random.seed(seed)
    fig = plt.figure()
    main()




