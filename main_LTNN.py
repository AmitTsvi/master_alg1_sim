import itertools

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


def plot_pegasos(h_array, s_array, codebook, train_dataset, test_dataset, m, n, test_n, d, inv_trans, lambda_scale=None):
    train_errors = []
    test_errors = []
    iteration_gap = 10
    train_true_classification = np.repeat(np.array([i for i in range(m)]), int(n/m), axis=0)
    test_true_classification = np.repeat(np.array([i for i in range(m)]), int(test_n/m), axis=0)
    for t in range(0, len(s_array), iteration_gap):
        train_classification = utils.decode_LTNN(codebook, train_dataset, m, n, d, h_array[t], s_array[t])
        test_classification = utils.decode_LTNN(codebook, test_dataset, m, test_n, d, h_array[t], s_array[t])
        train_errors.append(np.sum(train_classification != train_true_classification)/n)
        test_errors.append(np.sum(test_classification != test_true_classification)/test_n)
        if t % 200 == 0 and d == 2 and not lambda_sweep:
            utils.plot_decoding(train_dataset, train_classification, m, n, d, t)
    train_classification = utils.trans_decode(codebook, train_dataset, m, n, d, inv_trans)
    test_classification = utils.trans_decode(codebook, test_dataset, m, test_n, d, inv_trans)
    trans_train_error = np.sum(train_classification != train_true_classification)/n
    trans_test_error = np.sum(test_classification != test_true_classification)/test_n
    utils.plot_error_rate(train_errors, int(len(s_array)/iteration_gap)*[trans_train_error], test_errors,
                          int(len(s_array)/iteration_gap)*[trans_test_error], lambda_scale, iteration_gap)
    return train_errors, test_errors, trans_train_error, trans_test_error


def snr_test_plot(h, s, codebook, test_dataset, m, n, d, noise_type, noise_cov, mix_dist, snr_steps, org_energy, snr_seed, trans, inv_trans):
    np.random.seed(snr_seed)
    val_size = 4000
    datasets = []
    codebook_energy = np.mean(np.sum(np.power(codebook, 2), axis=1))
    snr_range = list(np.logspace(-2, np.log10(codebook_energy), 2*snr_steps))
    snr_range.append(org_energy)
    snr_range = list(np.sort(snr_range))
    for snr in snr_range:
        new_snr_dataset, _, _ = utils.gen_noise_dataset(noise_type, val_size, d, snr, noise_cov, mix_dist)
        new_snr_trans = utils.dataset_transform_LTNN(codebook, new_snr_dataset, m, val_size, d, trans)
        datasets.append(new_snr_trans)
    errors = []
    trans_errors = []
    true_classification = np.repeat(np.array([i for i in range(m)]), int(val_size/m), axis=0)
    print(snr_range)
    print(org_energy)
    print(codebook_energy)
    for index in range(len(snr_range)):
        print("SNR index " + str(index) + "\n")
        classification = utils.decode_LTNN(codebook, datasets[index], m, val_size, d, h, s)
        error = np.sum(classification != true_classification) / val_size
        errors.append(error)
        print(error)
        classification = utils.trans_decode(codebook, datasets[index], m, val_size, d, inv_trans)
        trans_error = np.sum(classification != true_classification) / val_size
        trans_errors.append(trans_error)
        print(trans_error)
        utils.plot_dataset(datasets[index], m, 10*np.log10(codebook_energy/snr_range[index]))
    utils.plot_snr_error_rate(errors, trans_errors, snr_range, org_energy, codebook_energy)


def subgradient_alg(iterations, m, n, etas, d_x, d_y, codebook, dataset, scale_lambda, partition, batch_size):
    s_array = []
    h_array = []
    s = np.zeros((d_x, d_x))
    h = np.zeros((d_y, d_x))
    print("Starting algorithm run with "+str(scale_lambda))
    for t in range(1, iterations+1):
        v_h_t = 0
        v_s_t = 0
        for k in range(batch_size):
            z_t = np.random.randint(n)
            y_t = np.expand_dims(dataset[z_t], axis=1)
            x_j = np.expand_dims(codebook[int(np.floor(z_t/(n/m)))], axis=1)
            for x_tag in codebook:
                x_tag_e = np.expand_dims(x_tag, axis=1)
                if not np.array_equal(x_tag_e, x_j):
                    indicate = y_t.T @ h @ (x_j - x_tag_e) - 0.5 * (x_j + x_tag_e).T @ s @ (x_j - x_tag_e)
                    if indicate < 1:
                        v_h_t += y_t @ (x_j-x_tag_e).T
                        v_s_t += (x_j+x_tag_e)@(x_j-x_tag_e).T+(x_j-x_tag_e)@(x_j+x_tag_e).T
        grad_h_t = 0
        grad_s_t = 0
        for i, p_i in enumerate(partition):
            delta_h = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(h, p_i.T), axis=0) ** 2)], axis=1)
            grad_h_t += etas[i] * 2 * (h @ delta_h @ delta_h.T)
            delta_s = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
            grad_s_t += etas[i] * (s @ delta_s @ delta_s.T + delta_s @ delta_s.T @ s)
        grad_h_t = scale_lambda[0]*grad_h_t - 3*v_h_t/(batch_size*(m-1))
        grad_s_t = scale_lambda[1]*grad_s_t + 3*0.25*v_s_t/(batch_size*(m-1))
        if scale_lambda[0] == 0 or scale_lambda[1] == 0:
            h -= (1/t)*grad_h_t
            s -= (1/t)*grad_s_t
        else:
            h -= (1/(scale_lambda[0]*t))*grad_h_t
            s -= (1/(scale_lambda[1]*t))*grad_s_t
        s = utils.get_near_psd(s)
        h_array.append(np.copy(h))
        s_array.append(np.copy(s))
    utils.plot_norms(h_array, s_array, scale_lambda)
    return h_array, s_array


def log_run_info(basic_dict):
    file1 = open("log.txt", "w")
    file1.write("d_x: "+str(basic_dict['d_x'])+" d_y:"+str(basic_dict['d_y'])+"\n")
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
    file1.write("Transformation type: "+basic_dict["trans_type"]+" with max singular value "+str(basic_dict["max_eigenvalue"]))
    file1.write("Batch size: "+str(basic_dict["batch_size"]))
    file1.close()


def save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset, h_array):
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
    f = open('h_array.npy', 'wb')
    np.save(f, h_array)
    f.close()


def main():
    if load:
        owd = os.getcwd()
        os.chdir("runs/LTNN")
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
        channel_trans, inv_channel_trans = utils.rebuild_trans_from_kernel(basic_dict['trans_kernel'],
                                                                           basic_dict['inv_trans_kernel'],
                                                                           basic_dict['trans_type'])
    else:
        utils.make_run_dir(load, None)
        d_x = 2
        d_y = 2
        basic_dict = {"d_x": d_x, "d_y": d_y, "m": 8, "n": 800, "test_n_ratio": 4, "iterations": 8000,
                      "scale_lambda": (0.002, 0.002),  "etas": (d_x+1)*[1/(d_x+1)], "seed": 9, "codebook_type": "Grid",
                      "codeword_energy": 1, "noise_type": "WhiteGaussian", "noise_energy": 0.02, "snr_steps": 10,
                      "snr_seed": 777, "trans_type": "Rotate", "max_eigenvalue": 0.01, "lambda_range": [-3, 0],
                      "batch_size": 1}
        np.random.seed(basic_dict['seed'])
        codebook, code_cov = utils.gen_codebook(basic_dict['codebook_type'], basic_dict['m'], basic_dict['d_x'])
        basic_dict['code_cov'] = code_cov
        channel_trans, inv_channel_trans, trans_kernel, inv_trans_kernel = utils.gen_transformation(basic_dict['d_x'],
                                                                                                    basic_dict['d_y'],
                                                                                                    basic_dict['trans_type'],
                                                                                                    basic_dict['max_eigenvalue'])
        basic_dict['trans_kernel'] = trans_kernel
        basic_dict['inv_trans_kernel'] = inv_trans_kernel
        noise_dataset, noise_cov, mix_dist = utils.gen_noise_dataset(basic_dict['noise_type'], basic_dict['n'],
                                                                     basic_dict['d_y'], basic_dict['noise_energy'])
        basic_dict['noise_cov'] = noise_cov
        basic_dict['mix_dist'] = mix_dist
        test_noise_dataset, _, _ = utils.gen_noise_dataset(basic_dict['noise_type'],
                                                           basic_dict["test_n_ratio"]*basic_dict['n'],
                                                           basic_dict['d_y'], basic_dict['noise_energy'],
                                                           noise_cov, mix_dist)
    train_dataset = utils.dataset_transform_LTNN(codebook, noise_dataset, basic_dict['m'], basic_dict['n'],
                                                 channel_trans)
    test_dataset = utils.dataset_transform_LTNN(codebook, test_noise_dataset, basic_dict['m'],
                                                basic_dict["test_n_ratio"]*basic_dict['n'], channel_trans)
    utils.plot_dataset(train_dataset, basic_dict['m'], 1/basic_dict['noise_energy'], codebook, inv_channel_trans)
    if not load_s_array:
        L = int(basic_dict['m'] * (basic_dict['m'] - 1) / 2)  # number of codewords pairs with i<j
        deltas = utils.delta_array(L, basic_dict['d_x'], basic_dict['m'], codebook)
        partition = utils.gen_partition(deltas)
        if lambda_sweep:
            log_range = np.logspace(basic_dict["lambda_range"][0], basic_dict["lambda_range"][1], 6)
            # for lambda_i in itertools.product(log_range, log_range):
            for lambda_i in log_range:
                h_array, s_array = subgradient_alg(basic_dict['iterations'], basic_dict['m'], basic_dict['n'],
                                                   basic_dict['etas'], basic_dict['d_x'], basic_dict['d_y'], codebook,
                                                   train_dataset, (lambda_i, lambda_i), partition, basic_dict["batch_size"])
                print("Finished running alg, now testing")
                _, _, _, _ = plot_pegasos(h_array, s_array, codebook, train_dataset, test_dataset, basic_dict['m'],
                                          basic_dict['n'], basic_dict['n'] * basic_dict['test_n_ratio'],
                                          basic_dict['d_y'], inv_channel_trans, lambda_i)
        else:
            h_array, s_array = subgradient_alg(basic_dict['iterations'], basic_dict['m'], basic_dict['n'],
                                               basic_dict['etas'],
                                               basic_dict['d_x'], basic_dict['d_y'], codebook, train_dataset,
                                               basic_dict['scale_lambda'], partition, basic_dict["batch_size"])
            print("Finished running alg, now testing")
    if load_errors:
        utils.plot_error_rate(basic_dict['train_errors'], basic_dict['iterations']*[basic_dict['cov_train_error']],
                              basic_dict['test_errors'], basic_dict['iterations']*[basic_dict['cov_test_error']])
    else:
        train_errors, test_errors, cov_train_error, cov_test_error = plot_pegasos(h_array, s_array, codebook,
                                                                                  train_dataset, test_dataset,
                                                                                  basic_dict['m'], basic_dict['n'],
                                                                                  basic_dict['n']*basic_dict['test_n_ratio'],
                                                                                  basic_dict['d_y'], inv_channel_trans)
        basic_dict['train_errors'] = train_errors
        basic_dict['test_errors'] = test_errors
        basic_dict['cov_train_error'] = cov_train_error
        basic_dict['cov_test_error'] = cov_test_error
    if snr_test:
        snr_test_plot(h_array[-1], s_array[-1], codebook, test_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d'],
                      basic_dict['noise_type'], basic_dict['noise_cov'], basic_dict['mix_dist'],
                      basic_dict['snr_steps'], basic_dict['noise_energy'], basic_dict["snr_seed"], channel_trans, inv_channel_trans)
    log_run_info(basic_dict)
    if just_replot_SNR:
        codebook_energy = np.mean(np.sum(np.power(codebook, 2), axis=1))
        utils.plot_snr_error_rate(errors, cov_errors, snr_range, basic_dict['noise_energy'], codebook_energy)
    if save:
        save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset, h_array)


if __name__ == '__main__':
    load = False
    load_s_array = False
    load_errors = False
    save = True
    snr_test = False
    just_replot_SNR = False
    lambda_sweep = False

    if snr_test:
        load = True
        load_s_array = True
        load_errors = True
    if just_replot_SNR:
        load = True
        load_s_array = True
        load_errors = True
    fig = plt.figure()
    main()
