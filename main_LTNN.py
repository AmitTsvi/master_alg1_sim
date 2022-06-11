import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from numpy import linalg as LA
import utils
import pickle
import os


def plot_pegasos(h_array, s_array, codebook, train_dataset, test_dataset, basic_dict, trans, lambda_scale=None):
    train_errors = []
    test_errors = []
    iteration_gap = 1
    test_n = basic_dict['n'] * basic_dict['test_n_ratio']
    train_true_classification = np.repeat(np.array([i for i in range(basic_dict['m'])]), int(basic_dict['n']/basic_dict['m']), axis=0)
    test_true_classification = np.repeat(np.array([i for i in range(basic_dict['m'])]), int(test_n/basic_dict['m']), axis=0)
    for t in range(0, len(s_array), iteration_gap):
        train_classification = utils.decode_LTNN(codebook, train_dataset, basic_dict['m'], basic_dict['n'], basic_dict['d_y'], h_array[t])
        test_classification = utils.decode_LTNN(codebook, test_dataset, basic_dict['m'], test_n, basic_dict['d_y'], h_array[t])
        train_errors.append(np.sum(train_classification != train_true_classification)/basic_dict['n'])
        test_errors.append(np.sum(test_classification != test_true_classification)/test_n)
        if t % 40 == 0 and basic_dict['d_y'] == 2 and not lambda_sweep:
            utils.plot_decoding(train_dataset, train_classification, basic_dict, t)
    train_classification = utils.trans_decode(codebook, train_dataset, trans)
    test_classification = utils.trans_decode(codebook, test_dataset, trans)
    trans_train_error = np.sum(train_classification != train_true_classification)/basic_dict['n']
    trans_test_error = np.sum(test_classification != test_true_classification)/test_n
    utils.plot_error_rate(train_errors, int(len(s_array)/iteration_gap)*[trans_train_error], test_errors,
                          int(len(s_array)/iteration_gap)*[trans_test_error], lambda_scale, iteration_gap)
    return train_errors, test_errors, trans_train_error, trans_test_error


def snr_test_plot(h, codebook, basic_dict, trans):
    np.random.seed(basic_dict["snr_seed"])
    val_size = 4000
    n_cycles = 20
    total_errors = np.zeros(len(basic_dict['snr_range']))
    total_trans_errors = np.zeros(len(basic_dict['snr_range']))
    noise_energy_range = [basic_dict['code_energy']*10**(-s/10) for s in basic_dict['snr_range']]
    for i in range(n_cycles):
        print("SNR Test Number "+str(i))
        datasets = []
        for n_energy in noise_energy_range:
            new_snr_dataset, _, _ = utils.gen_noise_dataset(basic_dict, val_size, basic_dict['noise_cov'], basic_dict['mix_dist'])
            new_snr_trans = utils.dataset_transform_LTNN(codebook, new_snr_dataset, basic_dict, val_size, trans)
            datasets.append(new_snr_trans)
        errors = np.zeros(len(basic_dict['snr_range']))
        trans_errors = np.zeros(len(basic_dict['snr_range']))
        true_classification = np.repeat(np.array([i for i in range(basic_dict['m'])]), int(val_size/basic_dict['m']), axis=0)
        # print(snr_range)
        # print(org_energy)
        for index in range(len(basic_dict['snr_range'])):
            print("SNR index " + str(index) + "\n")
            classification = utils.decode_LTNN(codebook, datasets[index], basic_dict['m'], val_size, basic_dict['d_y'], h)
            error = np.sum(classification != true_classification) / val_size
            errors[index] = error
            # print(error)
            classification = utils.trans_decode(codebook, datasets[index], trans)
            trans_error = np.sum(classification != true_classification) / val_size
            trans_errors[index] = trans_error
            if i == 0:
                utils.plot_dataset(datasets[index], basic_dict['snr_range'][index], codebook, basic_dict)
        total_errors += errors
        total_trans_errors += trans_errors
    total_errors = total_errors/n_cycles
    total_trans_errors = total_trans_errors/n_cycles
    utils.plot_snr_error_rate(total_errors, total_trans_errors, basic_dict)
    return total_errors, total_trans_errors


def subgradient_alg(basic_dict, codebook, dataset, scale_lambda, partition):
    s_array = []
    h_array = []
    s = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
    h = np.zeros((basic_dict['d_y'], basic_dict['d_x']))
    print("Starting algorithm run with "+str(scale_lambda))
    for t in range(1, basic_dict['iterations']+1):
        v_h_t = 0
        v_s_t = 0
        for k in range(basic_dict["batch_size"]):
            z_t = np.random.randint(basic_dict['n'])
            y_t = np.expand_dims(dataset[z_t], axis=1)
            x_j = np.expand_dims(codebook[int(np.floor(z_t/(basic_dict['n']/basic_dict['m'])))], axis=1)
            for x_tag in codebook:
                x_tag_e = np.expand_dims(x_tag, axis=1)
                if not np.array_equal(x_tag_e, x_j):
                    if basic_dict["with_s"]:
                        indicate = y_t.T @ h @ (x_j - x_tag_e) - 0.5 * (x_j + x_tag_e).T @ s @ (x_j - x_tag_e)
                    else:
                        indicate = y_t.T @ h @ (x_j - x_tag_e) - 0.5 * (x_j + x_tag_e).T @ h.T @ h @ (x_j - x_tag_e)
                    if indicate < 1:
                        if basic_dict["with_s"]:
                            v_h_t += y_t @ (x_j-x_tag_e).T
                            v_s_t += (x_j+x_tag_e)@(x_j-x_tag_e).T+(x_j-x_tag_e)@(x_j+x_tag_e).T
                        else:
                            v_h_t += y_t @ (x_j - x_tag_e).T-0.5*h@((x_j+x_tag_e)@(x_j-x_tag_e).T+(x_j-x_tag_e)@(x_j+x_tag_e).T)
        grad_h_t = 0
        grad_s_t = 0
        for i, p_i in enumerate(partition):
            delta_h = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(h, p_i.T), axis=0) ** 2)], axis=1)
            grad_h_t += basic_dict['etas'][i] * 2 * (h @ delta_h @ delta_h.T)
            delta_s = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
            grad_s_t += basic_dict['etas'][i] * (s @ delta_s @ delta_s.T + delta_s @ delta_s.T @ s)
        grad_h_t = scale_lambda[0]*grad_h_t - v_h_t/(basic_dict["batch_size"]*(basic_dict['m']-1))
        grad_s_t = scale_lambda[1]*grad_s_t + 0.25*v_s_t/(basic_dict["batch_size"]*(basic_dict['m']-1))
        if scale_lambda[0] == 0 or scale_lambda[1] == 0:
            h -= (1/t)*grad_h_t
            s -= (1/t)*grad_s_t
        else:
            h -= (1/(scale_lambda[0]*t))*grad_h_t
            s -= (1/(scale_lambda[1]*t))*grad_s_t
        if basic_dict["with_s"]:
            h, s = utils.projection(h, s)
        h_array.append(np.copy(h))
        s_array.append(np.copy(s))
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
    file1.write("Transformation Kernels:" + "\n")
    file1.write(str(basic_dict['trans_kernel']) + "\n")
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
            f = open('h_array.npy', 'rb')
            h_array = np.load(f)
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
        channel_trans = utils.rebuild_trans_from_kernel(basic_dict['trans_kernel'], basic_dict['trans_type'])
    else:
        d_x = 2
        d_y = 2
        basic_dict = {"d_x": d_x, "d_y": d_y, "m": 16, "n": 160, "test_n_ratio": 4, "iterations": 1600,
                      "scale_lambda": (0.06, 0.06),  "etas": (d_x+1)*[1/(d_x+1)], "seed": 3, "codebook_type": "Grid",
                      "codeword_energy": 1, "noise_type": "WhiteGaussian", "noise_energy": 0.01, "snr_steps": 10,
                      "snr_seed": 6, "trans_type": "Quadratic", "max_eigenvalue": 1, "min_eigenvalue": 0.8,
                      "lambda_range": [-1.4, -1.1], "batch_size": 1, "with_s": True, "model": "LTNN"}
        utils.make_run_dir(load, None, basic_dict)
        np.random.seed(basic_dict['seed'])
        codebook, code_cov = utils.gen_codebook(basic_dict)
        basic_dict['code_cov'] = code_cov
        basic_dict['code_energy'] = np.mean(np.sum(np.power(codebook, 2), axis=1))
        basic_dict['train_snr'] = 10*np.log10(basic_dict['code_energy']/basic_dict['noise_energy'])
        channel_trans, trans_kernel = utils.gen_transformation(basic_dict['d_x'], basic_dict['d_y'],
                                                               basic_dict['trans_type'],
                                                               basic_dict['max_eigenvalue'],
                                                               basic_dict['min_eigenvalue'])
        basic_dict['trans_kernel'] = trans_kernel
        noise_dataset, noise_cov, mix_dist = utils.gen_noise_dataset(basic_dict, basic_dict['n'])
        basic_dict['noise_cov'] = noise_cov
        basic_dict['mix_dist'] = mix_dist
        test_noise_dataset, _, _ = utils.gen_noise_dataset(basic_dict, basic_dict["test_n_ratio"]*basic_dict['n'],
                                                           noise_cov, mix_dist)
    train_dataset = utils.dataset_transform_LTNN(codebook, noise_dataset, basic_dict, basic_dict['n'],
                                                 channel_trans)
    test_dataset = utils.dataset_transform_LTNN(codebook, test_noise_dataset, basic_dict,
                                                basic_dict["test_n_ratio"]*basic_dict['n'], channel_trans)
    utils.plot_dataset(train_dataset, basic_dict['train_snr'], codebook, basic_dict)
    if not load_s_array:
        deltas = utils.delta_array(codebook, basic_dict)
        partition = utils.gen_partition(deltas)
        if lambda_sweep:
            log_range = np.logspace(basic_dict["lambda_range"][0], basic_dict["lambda_range"][1], 6)
            for lambda_i in log_range:
                h_array, s_array = subgradient_alg(basic_dict, codebook, train_dataset, (lambda_i, lambda_i), partition)
                print("Finished running alg, now testing")
                _, _, _, _ = plot_pegasos(h_array, s_array, codebook, train_dataset, test_dataset, basic_dict,
                                          channel_trans, lambda_i)
        else:
            h_array, s_array = subgradient_alg(basic_dict, codebook, train_dataset, basic_dict['scale_lambda'], partition)
            print("Finished running alg, now testing")
    if load_errors:
        utils.plot_error_rate(basic_dict['train_errors'], basic_dict['iterations']*[basic_dict['cov_train_error']],
                              basic_dict['test_errors'], basic_dict['iterations']*[basic_dict['cov_test_error']])
    else:
        train_errors, test_errors, cov_train_error, cov_test_error = plot_pegasos(h_array, s_array, codebook,
                                                                                  train_dataset, test_dataset,
                                                                                  basic_dict, channel_trans)
        basic_dict['train_errors'] = train_errors
        basic_dict['test_errors'] = test_errors
        basic_dict['cov_train_error'] = cov_train_error
        basic_dict['cov_test_error'] = cov_test_error
    snr_range = list(np.linspace(basic_dict['train_snr']-10, basic_dict['train_snr']+10, 2*basic_dict['snr_steps']))
    snr_range.append(basic_dict['train_snr'])
    basic_dict['snr_range'] = list(np.sort(snr_range))
    if snr_test:
        errors, trans_errors = snr_test_plot(h_array[-1], codebook, basic_dict, channel_trans)
        basic_dict['snr_errors'] = errors
        basic_dict['snr_trans_errors'] = trans_errors
    log_run_info(basic_dict)
    if just_replot_SNR:
        utils.plot_snr_error_rate(errors, cov_errors, basic_dict)
    if save:
        save_data(codebook, noise_dataset, s_array, basic_dict, test_noise_dataset, h_array)


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
