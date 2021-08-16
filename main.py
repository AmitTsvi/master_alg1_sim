import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from numpy.linalg import inv
from numpy import linalg as LA
import utils
import pickle
import os


def init_precision_matrix(d):
    a = np.random.rand(d, d)
    return np.dot(a, a.T)


def plot_pegasos(s_array, codebook, dataset, m, n, d):
    errors = []
    for t in range(len(s_array)):
        classification = utils.decode(codebook, dataset, m, n, d, s_array[t])
        true_classification = np.array([i for i in range(m) for j in range(n)])
        errors.append(np.sum(classification != true_classification)/(m*n))
        if t % 10 == 0 and d == 2:
            utils.plot_decoding(dataset, classification, m, n, d, t)
    utils.plot_error_rate(errors)
    return errors


def subgradient_alg(iterations, m, n, deltas, etas, d, codebook, dataset, scale_lambda, partition):
    s_array = []
    s = np.zeros((d, d))
    for t in range(1, iterations+1):
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
        s = utils.get_near_psd(s)
        s_array.append(s)
    return s_array


def log_run_info(basic_dict):
    file1 = open("log.txt", "w")
    file1.write("Dimension: "+str(basic_dict['d'])+"\n")
    file1.write("Number of codewords: " + str(basic_dict['m'])+"\n")
    file1.write("Codebook type: " + basic_dict['codebook_type'] + "\n")
    file1.write("codewords energy: " + str(basic_dict['codeword_energy'])+"\n")
    file1.write("Noise type: " + basic_dict['noise_type'] + "\n")
    file1.write("codewords energy: " + str(basic_dict['noise_energy']) + "\n")
    file1.write("Number of iterations: " + str(basic_dict['iterations'])+"\n")
    file1.write("Lambda: " + str(basic_dict['scale_lambda'])+"\n")
    file1.write("Etas: " + str(basic_dict['etas'])+"\n")
    file1.write("Seed: " + str(basic_dict['seed']) + "\n")
    if basic_dict['code_cov'] is not None:
        file1.write("Random codebook covariance:" + "\n")
        file1.write(str(basic_dict['code_cov']) + "\n")
    if basic_dict['noise_cov'] is not None:
        file1.write("Random noise covariance(s):" + "\n")
        file1.write(str(basic_dict['noise_cov']) + "\n")
    file1.close()


def save_data(codebook, noise_dataset, s_array, basic_dict):
    f = open('codebook.npy', 'wb')
    np.save(f, codebook)
    f.close()
    f = open('noise_dataset.npy', 'wb')
    np.save(f, noise_dataset)
    f.close()
    f = open('s_array.npy', 'wb')
    np.save(f, s_array)
    f.close()
    outfile = open("basic_dict", 'wb')
    pickle.dump(basic_dict, outfile)
    outfile.close()


def main():
    if load:
        owd = os.getcwd()
        os.chdir("runs")
        workdir = input("Insert load dir: ")
        os.chdir(workdir)
        infile = open('basic_dict', 'rb')
        basic_dict = pickle.load(infile)
        infile.close()
        f = open('codebook.npy', 'rb')
        codebook = np.load(f)
        f.close()
        f = open('noise_dataset.npy', 'rb')
        noise_dataset = np.load(f)
        f.close()
        np.random.seed(basic_dict['seed'])
        if load_s_array:
            f = open('s_array.npy', 'rb')
            s_array = np.load(f)
            f.close()
        os.chdir(owd)
        utils.make_run_dir()
    else:
        utils.make_run_dir()
        basic_dict = {"d": 3, "m": 124, "n": 100, "iterations": 100, "scale_lambda": 0.1, "etas": 3*[0.5], "seed": 8,
                      "codebook_type": "Grid", "codeword_energy": 1, "noise_type": "Gaussian", "noise_energy": 0.1}
        np.random.seed(basic_dict['seed'])
        codebook, code_cov = utils.gen_codebook(basic_dict['codebook_type'], basic_dict['m'], basic_dict['d'])
        noise_dataset, noise_cov = utils.gen_noise_dataset(basic_dict['noise_type'], basic_dict['n'], basic_dict['d'],
                                                           basic_dict['noise_energy'])
        basic_dict['code_cov'] = code_cov
        basic_dict['noise_cov'] = noise_cov
    dataset = utils.dataset_transform(codebook, noise_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d'])
    utils.plot_dataset(dataset, basic_dict['m'], fig)
    if not load_s_array:
        L = int(basic_dict['m'] * (basic_dict['m'] - 1) / 2)  # number of codewords pairs with i<j
        deltas = utils.delta_array(L, basic_dict['d'], basic_dict['m'], codebook)
        partition = utils.gen_partition(basic_dict['d'], deltas)
        s_array = subgradient_alg(basic_dict['iterations'], basic_dict['m'], basic_dict['n'], deltas, basic_dict['etas'],
                                  basic_dict['d'], codebook, dataset, basic_dict['scale_lambda'], partition)
    errors = plot_pegasos(s_array, codebook, dataset, basic_dict['m'], basic_dict['n'], basic_dict['d'])
    basic_dict['errors'] = errors
    log_run_info(basic_dict)
    if save:
        save_data(codebook, noise_dataset, s_array, basic_dict)


if __name__ == '__main__':
    load = False
    load_s_array = False
    save = True
    fig = plt.figure()
    main()
