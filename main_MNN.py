import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from numpy import linalg as LA
import utils
import pickle
import os


def plot_pegasos(s_array, codebook, train_dataset, test_dataset, basic_dict, lambda_scale=None):
    iteration_gap = 20
    train_errors = []
    test_errors = []
    test_n = basic_dict['n'] * basic_dict['test_n_ratio']
    train_true_classification = np.array([i for i in range(basic_dict['m']) for j in range(basic_dict['n'])])
    test_true_classification = np.array([i for i in range(basic_dict['m']) for j in range(test_n)])
    for t in range(0, len(s_array), iteration_gap):
        train_classification = utils.decode(codebook, train_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d_x'], s_array[t])
        test_classification = utils.decode(codebook, test_dataset, basic_dict['m'], test_n, basic_dict['d_x'], s_array[t])
        train_errors.append(np.sum(train_classification != train_true_classification)/(basic_dict['m']*basic_dict['n']))
        test_errors.append(np.sum(test_classification != test_true_classification) / (test_n*basic_dict['m']))
        if t % 10 == 0 and basic_dict['d_x'] == 2:
            utils.plot_decoding(train_dataset, train_classification, basic_dict, t)
    if len(basic_dict['noise_cov'].shape) == 2:
        precision = LA.inv(basic_dict['noise_cov'])
    else:
        precision = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
        for i in range(len(basic_dict['mix_dist'])):
            precision += basic_dict['mix_dist'][i]*basic_dict['noise_cov'][i]
        precision = LA.inv(precision)
    train_classification = utils.decode(codebook, train_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d_x'], precision)
    test_classification = utils.decode(codebook, test_dataset, basic_dict['m'], test_n, basic_dict['d_x'], precision)
    cov_train_error = np.sum(train_classification != train_true_classification)/(basic_dict['m']*basic_dict['n'])
    cov_test_error = np.sum(test_classification != test_true_classification) / (test_n*basic_dict['m'])
    utils.plot_error_rate(train_errors, int(len(s_array)/iteration_gap)*[cov_train_error], test_errors,
                          int(len(s_array)/iteration_gap)*[cov_test_error], lambda_scale, iteration_gap)
    return train_errors, test_errors, cov_train_error, cov_test_error


def snr_test_plot(s, codebook, basic_dict):
    np.random.seed(777)
    val_size = 5000
    n_cycles = 20
    total_errors = np.zeros(len(basic_dict['snr_range']))
    total_trans_errors = np.zeros(len(basic_dict['snr_range']))
    total_mean_sol_errors = np.zeros(len(basic_dict['snr_range']))
    noise_energy_range = [basic_dict['code_energy']*10**(-s/10) for s in basic_dict['snr_range']]
    if len(basic_dict['noise_cov'].shape) == 2:
        precision = LA.inv(basic_dict['noise_cov'])
    else:
        precision = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
        for i in range(len(basic_dict['mix_dist'])):
            precision += basic_dict['mix_dist'][i]*basic_dict['noise_cov'][i]
        precision = LA.inv(precision)
    for i in range(n_cycles):
        print("SNR Test Number "+str(i))
        datasets = []
        for n_energy in noise_energy_range:
            new_snr_dataset, _, _ = utils.gen_noise_dataset(basic_dict, val_size, basic_dict['noise_cov'], basic_dict['mix_dist'], n_energy)
            new_snr_trans = utils.dataset_transform(codebook, new_snr_dataset, val_size, basic_dict)
            datasets.append(new_snr_trans)
        errors = np.zeros(len(basic_dict['snr_range']))
        cov_errors = np.zeros(len(basic_dict['snr_range']))
        mean_sol_errors = np.zeros(len(basic_dict['snr_range']))
        true_classification = np.array([i for i in range(basic_dict['m']) for j in range(val_size)])
        for index in range(len(basic_dict['snr_range'])):
            print("SNR index " + str(index) + "\n")
            classification = utils.decode(codebook, datasets[index], basic_dict['m'], val_size, basic_dict['d_x'], s)
            error = np.sum(classification != true_classification) / (val_size*basic_dict['m'])
            errors[index] = error
            # print(error)
            classification = utils.decode(codebook, datasets[index], basic_dict['m'], val_size, basic_dict['d_x'], precision)
            cov_error = np.sum(classification != true_classification) / (val_size*basic_dict['m'])
            cov_errors[index] = cov_error
            classification = utils.decode(codebook, datasets[index], basic_dict['m'], val_size, basic_dict['d_x'], np.eye(basic_dict['d_x']))
            mean_sol_error = np.sum(classification != true_classification) / val_size
            mean_sol_errors[index] = mean_sol_error
            if i == 0:
                utils.plot_dataset(datasets[index], basic_dict['snr_range'][index], codebook, basic_dict)
        total_errors += errors
        total_trans_errors += cov_errors
        total_mean_sol_errors += mean_sol_errors
    total_errors = total_errors/n_cycles
    total_trans_errors = total_trans_errors/n_cycles
    total_mean_sol_errors = total_mean_sol_errors / n_cycles
    utils.plot_snr_error_rate(total_errors, total_trans_errors, basic_dict, total_mean_sol_errors)
    return total_errors, total_trans_errors


def subgradient_alg(basic_dict, deltas, codebook, dataset, scale_lambda, partition):
    s_array = []
    s = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
    for t in range(1, basic_dict['iterations']+1):
        p_t = np.random.randint(basic_dict['m']-1)
        q_t = np.random.randint(p_t+1, basic_dict['m'])
        x_p_t = np.expand_dims(codebook[p_t], axis=1)
        x_q_t = np.expand_dims(codebook[q_t], axis=1)
        z_t = np.random.randint(basic_dict['n'])
        which_word = np.random.randint(2)
        y_t = np.expand_dims(dataset[p_t, z_t] if which_word == 0 else dataset[q_t, z_t], axis=1)
        delta_p_q_t = np.expand_dims(deltas[utils.double_to_single_index(p_t, q_t, basic_dict['m'])], axis=1)
        a_t = ((-1)**which_word)*(y_t-0.5*(x_p_t+x_q_t))
        v_t = 0
        if a_t.T@s@delta_p_q_t < 1:
            v_t = 0.5*(delta_p_q_t@a_t.T+a_t@delta_p_q_t.T)
        grad_t = 0
        for i, p_i in enumerate(partition):
            delta_p_q_star = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
            grad_t += basic_dict['etas'][i] * (s @ delta_p_q_star @ delta_p_q_star.T + delta_p_q_star @ delta_p_q_star.T @ s)
        grad_t = scale_lambda*grad_t - v_t
        s -= (1/(scale_lambda*t))*grad_t
        s = utils.get_near_psd(s)
        s_array.append(s)
    return s_array


def log_run_info(basic_dict):
    file1 = open("log.txt", "w")
    file1.write("Dimension: "+str(basic_dict['d_x'])+"\n")
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
        os.chdir("runs/MNN")
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
        utils.make_run_dir(load, workdir, basic_dict)
    else:
        d = 4
        basic_dict = {"d_x": d, "d_y": d, "m": 256, "n": 32, "test_n_ratio": 4, "iterations": 800,
                      "scale_lambda": 0.1, "etas": (d+1)*[1/(d+1)], "seed": 61, "codebook_type": "Grid",
                      "codeword_energy": 1, "noise_type": "Mixture", "noise_energy": 0.05, "snr_steps": 10,
                      "snr_seed": 6, "lambda_range": [-2, -1], "batch_size": 1, "model": "MNN"}
        utils.make_run_dir(load, None, basic_dict)
        np.random.seed(basic_dict['seed'])
        codebook, code_cov = utils.gen_codebook(basic_dict)
        basic_dict['code_cov'] = code_cov
        basic_dict['code_energy'] = np.mean(np.sum(np.power(codebook, 2), axis=1))
        basic_dict['train_snr'] = 10*np.log10(basic_dict['code_energy']/basic_dict['noise_energy'])
        noise_dataset, noise_cov, mix_dist = utils.gen_noise_dataset(basic_dict, basic_dict['n'])
        basic_dict['noise_cov'] = noise_cov
        basic_dict['mix_dist'] = mix_dist
        test_noise_dataset, _, _ = utils.gen_noise_dataset(basic_dict, 4*basic_dict['n'], noise_cov, mix_dist)
    dataset = utils.dataset_transform(codebook, noise_dataset, basic_dict['n'], basic_dict)
    test_dataset = utils.dataset_transform(codebook, test_noise_dataset, 4*basic_dict['n'], basic_dict)
    basic_dict['mean_sol'] = utils.mean_solution(basic_dict, dataset)
    utils.plot_dataset(dataset, basic_dict['train_snr'], codebook, basic_dict)
    if not load_s_array:
        deltas = utils.delta_array(codebook, basic_dict)
        partition = utils.gen_partition(codebook)
        if lambda_sweep:
            log_range = np.logspace(basic_dict["lambda_range"][0], basic_dict["lambda_range"][1], 6)
            for lambda_i in log_range:
                s_array = subgradient_alg(basic_dict, deltas, codebook, dataset, lambda_i, partition)
                print("Finished running alg, now testing")
                _, _, _, _ = plot_pegasos(s_array, codebook, dataset, test_dataset, basic_dict, lambda_i)
        else:
            s_array = subgradient_alg(basic_dict, deltas, codebook, dataset, basic_dict['scale_lambda'], partition)
            print("Finished running alg, now testing")
    if load_errors:
        utils.plot_error_rate(basic_dict['train_errors'], len(basic_dict['train_errors'])*[basic_dict['cov_train_error']],
                              basic_dict['test_errors'],  len(basic_dict['test_errors'])*[basic_dict['cov_test_error']])
    else:
        train_errors, test_errors, cov_train_error, cov_test_error = plot_pegasos(s_array, codebook, dataset,
                                                                                  test_dataset, basic_dict)
        basic_dict['train_errors'] = train_errors
        basic_dict['test_errors'] = test_errors
        basic_dict['cov_train_error'] = cov_train_error
        basic_dict['cov_test_error'] = cov_test_error
    snr_range = list(np.linspace(basic_dict['train_snr']-10, basic_dict['train_snr']+10, 2*basic_dict['snr_steps']))
    snr_range.append(basic_dict['train_snr'])
    basic_dict['snr_range'] = list(np.sort(snr_range))
    if snr_test:
        errors, cov_errors = snr_test_plot(s_array[-1], codebook, basic_dict)
        basic_dict['snr_errors'] = errors
        basic_dict['snr_cov_errors'] = cov_errors
    log_run_info(basic_dict)
    if just_replot_SNR:
        utils.plot_snr_error_rate(errors, cov_errors, basic_dict)
    if save:
        save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset)


if __name__ == '__main__':
    load = False
    load_s_array = False
    load_errors = False
    save = True
    snr_test = True
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
