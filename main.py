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


def plot_pegasos(s_array, codebook, train_dataset, test_dataset, m, n, d, noise_cov, mix_dist):
    train_errors = []
    test_errors = []
    train_true_classification = np.array([i for i in range(m) for j in range(n)])
    test_true_classification = np.array([i for i in range(m) for j in range(4 * n)])
    for t in range(len(s_array)):
        train_classification = utils.decode(codebook, train_dataset, m, n, d, s_array[t])
        test_classification = utils.decode(codebook, test_dataset, m, 4*n, d, s_array[t])
        train_errors.append(np.sum(train_classification != train_true_classification)/(m*n))
        test_errors.append(np.sum(test_classification != test_true_classification) / (4*m*n))
        if t % 10 == 0 and d == 2:
            utils.plot_decoding(train_dataset, train_classification, m, n, d, t)
    if len(noise_cov.shape) == 2:
        precision = LA.inv(noise_cov)
    else:
        precision = np.zeros((d, d))
        for i in range(len(mix_dist)):
            precision += mix_dist[i]*noise_cov[i]
        precision = LA.inv(precision)
    train_classification = utils.decode(codebook, train_dataset, m, n, d, precision)
    test_classification = utils.decode(codebook, test_dataset, m, 4*n, d, precision)
    cov_train_error = np.sum(train_classification != train_true_classification)/(m*n)
    cov_test_error = np.sum(test_classification != test_true_classification) / (4*m*n)
    utils.plot_error_rate(train_errors, len(s_array)*[cov_train_error], test_errors, len(s_array)*[cov_test_error])
    return train_errors, test_errors, cov_train_error, cov_test_error


def snr_test_plot(s, codebook, test_dataset, m, n, d, noise_type, noise_cov, mix_dist, snr_steps, org_energy):
    np.random.seed(777)
    val_size = 10000
    datasets = []
    codebook_energy = np.mean(np.sum(np.power(codebook, 2), axis=1))
    snr_range = list(np.logspace(-2, np.log10(codebook_energy), 2*snr_steps))
    snr_range.append(org_energy)
    snr_range = list(np.sort(snr_range))
    for snr in snr_range:
        new_snr_dataset, _, _ = utils.gen_noise_dataset(noise_type, val_size, d, snr, noise_cov, mix_dist)
        new_snr_trans = utils.dataset_transform(codebook, new_snr_dataset, m, val_size, d)
        datasets.append(new_snr_trans)
    errors = []
    cov_errors = []
    true_classification = np.array([i for i in range(m) for j in range(val_size)])
    if len(noise_cov.shape) == 2:
        precision = LA.inv(noise_cov)
    else:
        precision = np.zeros((d, d))
        for i in range(len(mix_dist)):
            precision += mix_dist[i]*noise_cov[i]
        precision = LA.inv(precision)
    print(snr_range)
    print(org_energy)
    print(codebook_energy)
    for index in range(len(snr_range)):
        print("SNR index " + str(index) + "\n")
        classification = utils.decode(codebook, datasets[index], m, val_size, d, s)
        error = np.sum(classification != true_classification) / (val_size*m)
        errors.append(error)
        print(error)
        classification = utils.decode(codebook, datasets[index], m, val_size, d, precision)
        cov_error = np.sum(classification != true_classification) / (val_size*m)
        cov_errors.append(cov_error)
        print(cov_error)
        utils.plot_dataset(datasets[index], m, 10*np.log10(codebook_energy/snr_range[index]))
    utils.plot_snr_error_rate(errors, cov_errors, snr_range, org_energy, codebook_energy)


def subgradient_alg(iterations, m, n, deltas, etas, d, codebook, dataset, scale_lambda, partition):
    s_array = []
    s = np.zeros((d, d))
    for t in range(1, iterations+1):
        print("Iteration "+str(t)+"\n")
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
    file1.write("Number of noise samples: " + str(basic_dict['n']) + "\n")
    file1.write("Codebook type: " + basic_dict['codebook_type'] + "\n")
    file1.write("Codewords maximal energy: " + str(basic_dict['codeword_energy'])+"\n")
    file1.write("Noise type: " + basic_dict['noise_type'] + "\n")
    file1.write("Noise energy: " + str(basic_dict['noise_energy']) + "\n")
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
    if basic_dict['mix_dist'] is not None:
        file1.write("Gaussian mixture distribution:" + "\n")
        file1.write(str(basic_dict['mix_dist']) + "\n")
    file1.close()


def save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset):
    f = open('codebook.npy', 'wb')
    np.save(f, codebook)
    f.close()
    f = open('noise_dataset.npy', 'wb')
    np.save(f, noise_dataset)
    f.close()
    f = open('test_noise_dataset.npy', 'wb')
    np.save(f, test_noise_dataset)
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
        org_workdir = workdir
        if just_replot_SNR:
            workdir = workdir.split("load_")[1]
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
        f = open('test_noise_dataset.npy', 'rb')
        test_noise_dataset = np.load(f)
        f.close()
        np.random.seed(basic_dict['seed'])
        if load_s_array:
            f = open('s_array.npy', 'rb')
            s_array = np.load(f)
            f.close()
        if just_replot_SNR:
            os.chdir("..")
            os.chdir(org_workdir)
            f = open('SNR_errors.npy', 'rb')
            errors = np.load(f)
            f.close()
            f = open('SNR_cov_errors.npy', 'rb')
            cov_errors = np.load(f)
            f.close()
            f = open('SNR_range.npy', 'rb')
            snr_range = np.load(f)
            f.close()
        os.chdir(owd)
        utils.make_run_dir(load, workdir)
    else:
        utils.make_run_dir(load, None)
        basic_dict = {"d": 4, "m": 256, "n": 100, "iterations": 100, "scale_lambda": 0.1, "etas": 4*[0.25], "seed": 61,
                      "codebook_type": "Grid", "codeword_energy": 1, "noise_type": "Mixture",
                      "noise_energy": 0.05, "snr_steps": 10}
        np.random.seed(basic_dict['seed'])
        codebook, code_cov = utils.gen_codebook(basic_dict['codebook_type'], basic_dict['m'], basic_dict['d'])
        basic_dict['code_cov'] = code_cov
        noise_dataset, noise_cov, mix_dist = utils.gen_noise_dataset(basic_dict['noise_type'], basic_dict['n'],
                                                                     basic_dict['d'], basic_dict['noise_energy'])
        basic_dict['noise_cov'] = noise_cov
        basic_dict['mix_dist'] = mix_dist
        test_noise_dataset, _, _ = utils.gen_noise_dataset(basic_dict['noise_type'], 4*basic_dict['n'], basic_dict['d'],
                                                           basic_dict['noise_energy'], noise_cov, mix_dist)
    dataset = utils.dataset_transform(codebook, noise_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d'])
    test_dataset = utils.dataset_transform(codebook, test_noise_dataset, basic_dict['m'], 4*basic_dict['n'],
                                           basic_dict['d'])
    utils.plot_dataset(dataset, basic_dict['m'], 1/basic_dict['noise_energy'])
    if not load_s_array:
        L = int(basic_dict['m'] * (basic_dict['m'] - 1) / 2)  # number of codewords pairs with i<j
        deltas = utils.delta_array(L, basic_dict['d'], basic_dict['m'], codebook)
        partition = utils.gen_partition(basic_dict['d'], deltas)
        s_array = subgradient_alg(basic_dict['iterations'], basic_dict['m'], basic_dict['n'], deltas, basic_dict['etas'],
                                  basic_dict['d'], codebook, dataset, basic_dict['scale_lambda'], partition)
        print("Finished running alg, now testing")
    if load_errors:
        utils.plot_error_rate(basic_dict['train_errors'], basic_dict['iterations']*[basic_dict['cov_train_error']],
                              basic_dict['test_errors'], basic_dict['iterations']*[basic_dict['cov_test_error']])
    else:
        train_errors, test_errors, cov_train_error, cov_test_error = plot_pegasos(s_array, codebook, dataset,
                                                                                  test_dataset, basic_dict['m'],
                                                                                  basic_dict['n'], basic_dict['d'],
                                                                                  basic_dict['noise_cov'],
                                                                                  basic_dict['mix_dist'])
        basic_dict['train_errors'] = train_errors
        basic_dict['test_errors'] = test_errors
        basic_dict['cov_train_error'] = cov_train_error
        basic_dict['cov_test_error'] = cov_test_error
    if snr_test:
        snr_test_plot(s_array[-1], codebook, test_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d'],
                      basic_dict['noise_type'], basic_dict['noise_cov'], basic_dict['mix_dist'],
                      basic_dict['snr_steps'], basic_dict['noise_energy'])
    log_run_info(basic_dict)
    if just_replot_SNR:
        codebook_energy = np.mean(np.sum(np.power(codebook, 2), axis=1))
        utils.plot_snr_error_rate(errors, cov_errors, snr_range, basic_dict['noise_energy'], codebook_energy)
    if save:
        save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset)


if __name__ == '__main__':
    load = False
    load_s_array = False
    load_errors = False
    save = False
    snr_test = False
    if snr_test:
        load = True
        load_s_array = True
        load_errors = True
    just_replot_SNR = True
    if just_replot_SNR:
        load = True
        load_s_array = True
        load_errors = True
    fig = plt.figure()
    main()
