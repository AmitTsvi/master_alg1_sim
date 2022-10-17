import os
import pickle
import numpy as np
import utils
from abc import ABC, abstractmethod
from datetime import datetime


class CommChannel(ABC):

    def __init__(self, model):
        self.workdir = "runs/" + model
        self.load = False
        self.load_sol_array = False
        self.load_errors = False
        self.save = False
        self.snr_test = False
        self.just_replot_SNR = False
        self.lambda_sweep = False
        self.labels = None
        self.init_operation_type()

    def init_operation_type(self):
        op_num = input("Insert operation type:\n1: save\n2: Lambda sweep\n3: SNR test\nYour choice: ")
        if op_num == '1':
            self.save = True
        elif op_num == '2':
            self.lambda_sweep = True
        elif op_num == '3':
            self.load = True
            self.load_sol_array = True
            self.load_errors = True
            self.snr_test = True
        else:
            print("Illegal choice")
            exit()

    def load_data(self):
        os.chdir(self.workdir)
        saved_dir = input("Insert load directory: ")
        if self.just_replot_SNR:
            saved_dir = saved_dir.split("load_")[1]
        os.chdir(saved_dir)
        infile = open('basic_dict', 'rb')
        basic_dict = pickle.load(infile)
        infile.close()
        f = open('codebook.npy', 'rb')
        codebook = np.load(f)
        f.close()
        f = open('train_noise_dataset.npy', 'rb')
        train_noise_dataset = np.load(f)
        f.close()
        f = open('test_noise_dataset.npy', 'rb')
        test_noise_dataset = np.load(f)
        f.close()
        solution = None
        if self.load_sol_array:
            solution = self.load_sol()
        return saved_dir, basic_dict, codebook, train_noise_dataset, test_noise_dataset, solution

    @abstractmethod
    def init_dict(self):
        pass

    @abstractmethod
    def load_sol(self):
        pass

    def gen_codebook(self, basic_dict):
        codebook, code_cov = utils.gen_codebook(basic_dict)
        basic_dict['code_cov'] = code_cov
        basic_dict['code_energy'] = np.mean(np.sum(np.power(codebook, 2), axis=1))
        basic_dict['train_snr'] = 10*np.log10(basic_dict['code_energy']/basic_dict['noise_energy'])
        return codebook

    def gen_noise_datasets(self, basic_dict):
        train_noise_dataset, noise_cov, mix_means, mix_dist, df = utils.gen_noise_dataset(basic_dict, basic_dict['n'])
        basic_dict['noise_cov'] = noise_cov
        basic_dict['mix_dist'] = mix_dist
        basic_dict['mix_means'] = mix_means
        basic_dict['df'] = df
        test_noise_dataset, _, _, _, _ = utils.gen_noise_dataset(basic_dict, basic_dict["test_n_ratio"]*basic_dict['n'],
                                                           noise_cov, mix_means, mix_dist, None, df)
        return train_noise_dataset, test_noise_dataset

    @abstractmethod
    def get_rule(self, basic_dict):
        pass

    @abstractmethod
    def transform_dataset(self, codebook, noise_dataset, basic_dict, trans=None):
        pass

    def run_algorithm(self, basic_dict, codebook, train_dataset, test_dataset):
        deltas = utils.delta_array(codebook, basic_dict)
        partition = utils.gen_partition(deltas)
        if self.lambda_sweep:
            log_range = np.logspace(basic_dict["lambda_range"][0], basic_dict["lambda_range"][1], 12)
            for lambda_i in log_range:
                solution, obj_vals = self.subgradient_alg(basic_dict, codebook, train_dataset, lambda_i, partition, deltas)
                print("Finished running alg, now testing")
                self.plot_pegasos(solution[0], codebook, train_dataset, test_dataset, basic_dict, obj_vals, lambda_i)
        else:
            solution, obj_vals = self.subgradient_alg(basic_dict, codebook, train_dataset, basic_dict['scale_lambda'], partition, deltas)
            print("Finished running alg, now testing")
            self.plot_pegasos(solution[0], codebook, train_dataset, test_dataset, basic_dict, obj_vals, basic_dict['scale_lambda'])
        return solution, obj_vals

    @abstractmethod
    def subgradient_alg(self, basic_dict, codebook, dataset, lambda_i, partition, deltas):
        pass

    def plot_pegasos(self, decoders, codebook, train_dataset, test_dataset, basic_dict, obj_vals, lambda_scale=None):
        train_errors = []
        test_errors = []
        test_n = basic_dict['n'] * basic_dict['test_n_ratio']
        train_true_classification = self.get_true_classification(basic_dict, basic_dict['n'])
        test_true_classification = self.get_true_classification(basic_dict, test_n)
        for t in range(0, basic_dict['iterations'], basic_dict['iter_gap']):
            train_classification = self.decode(codebook, train_dataset, decoders[t])
            test_classification = self.decode(codebook, test_dataset, decoders[t])
            train_errors.append(np.sum(train_classification != train_true_classification) / len(train_true_classification))
            test_errors.append(np.sum(test_classification != test_true_classification) / len(test_true_classification))
            if t % 50 == 0 and basic_dict['d_x'] == 2 and not self.lambda_sweep:
                utils.plot_decoding(train_dataset, train_classification, basic_dict, t)
        rule = self.get_rule(basic_dict)
        train_rule_classification = self.rule_decode(codebook, train_dataset, rule)
        test_rule_classification = self.rule_decode(codebook, test_dataset, rule)
        train_rule_error = np.sum(train_rule_classification != train_true_classification) / len(train_true_classification)
        test_rule_error = np.sum(test_rule_classification != test_true_classification) / len(test_true_classification)
        train_estimator_classification = self.estimator_decode(codebook, train_dataset, basic_dict['estimation_decoder'])
        test_estimator_classification = self.estimator_decode(codebook, test_dataset, basic_dict['estimation_decoder'])
        train_estimator_error = np.sum(train_estimator_classification != train_true_classification) / len(train_true_classification)
        test_estimator_error = np.sum(test_estimator_classification != test_true_classification) / len(test_true_classification)
        utils.plot_error_rate(train_errors, train_rule_error, train_estimator_error, test_errors, test_rule_error,
                              test_estimator_error, obj_vals, lambda_scale, basic_dict['iter_gap'])
        basic_dict['train_errors'] = train_errors
        basic_dict['test_errors'] = test_errors
        basic_dict['train_rule_error'] = train_rule_error
        basic_dict['test_rule_error'] = test_rule_error
        basic_dict['train_estimator_error'] = train_estimator_error
        basic_dict['test_estimator_error'] = test_estimator_error
        basic_dict['obj_vals'] = obj_vals
        basic_dict['min_test_iter'] = 50+np.argmin(test_errors[50:])

    @abstractmethod
    def get_true_classification(self, basic_dict, test_n):
        pass

    @abstractmethod
    def decode(self, codebook, dataset, decoder):
        pass

    @abstractmethod
    def rule_decode(self, codebook, dataset, rule):
        pass

    @abstractmethod
    def no_learning_decode(self, codebook, dataset):
        pass

    @abstractmethod
    def get_estimation_decoder(self, codebook, dataset, noise_dataset):
        pass

    @abstractmethod
    def estimator_decode(self, codebook, dataset, estimator):
        pass

    def run_snr_test(self, basic_dict, codebook, solution):
        snr_range = list(np.linspace(basic_dict['train_snr'] - 10, basic_dict['train_snr'] + 10, 2 * basic_dict['snr_steps']))
        snr_range.append(basic_dict['train_snr'])
        basic_dict['snr_range'] = list(np.sort(snr_range))
        all_errors = self.perform_snr_test(solution[0][basic_dict['min_test_iter']], codebook, basic_dict)
        basic_dict['snr_errors'], basic_dict['snr_rule_errors'], basic_dict['snr_estimation_errors'], basic_dict['snr_naive_errors'] = all_errors

    def perform_snr_test(self, decoder, codebook, basic_dict):
        np.random.seed(basic_dict["snr_seed"])
        val_size = basic_dict["snr_val_size"]
        n_cycles = basic_dict["snr_test_cycles"]
        total_errors = np.zeros(len(basic_dict['snr_range']))
        total_rule_errors = np.zeros(len(basic_dict['snr_range']))
        total_estimator_errors = np.zeros(len(basic_dict['snr_range']))
        total_no_learning_errors = np.zeros(len(basic_dict['snr_range']))
        noise_energy_range = [basic_dict['code_energy'] * 10 ** (-s / 10) for s in basic_dict['snr_range']]
        rule = self.get_rule(basic_dict)
        true_classification = self.get_true_classification(basic_dict, val_size)
        for i in range(n_cycles):
            now = datetime.now()
            print("SNR Test Number " + str(i) + " Started at " + now.strftime("%Y_%m_%d_%H%M%S"))
            datasets = []
            for n_energy in noise_energy_range:
                new_snr_dataset, _, _, _, _ = utils.gen_noise_dataset(basic_dict, val_size, basic_dict['noise_cov'],
                                                                      basic_dict['mix_means'], basic_dict['mix_dist'],
                                                                      n_energy, basic_dict['df'])
                new_snr_trans = self.transform_dataset(codebook, new_snr_dataset, basic_dict, rule)
                datasets.append(new_snr_trans)
            for index in range(len(basic_dict['snr_range'])):
                print("SNR index " + str(index))
                learn_classification = self.decode(codebook, datasets[index], decoder)
                total_errors[index] += np.sum(learn_classification != true_classification) / len(true_classification)
                rule_classification = self.rule_decode(codebook, datasets[index], rule)
                total_rule_errors[index] += np.sum(rule_classification != true_classification) / len(true_classification)
                estimate_classification = self.estimator_decode(codebook, datasets[index], basic_dict['estimation_decoder'])
                total_estimator_errors[index] += np.sum(estimate_classification != true_classification) / len(true_classification)
                no_learn_classification = self.no_learning_decode(codebook, datasets[index])
                if no_learn_classification.all() is not None:
                    total_no_learning_errors[index] += np.sum(no_learn_classification != true_classification) / len(true_classification)
                if i == 0:
                    utils.plot_dataset(datasets[index], basic_dict['snr_range'][index], codebook, basic_dict)
                if i == 0 and basic_dict['d_x'] == 2:
                    utils.plot_decoding(datasets[index], learn_classification, basic_dict, 1, "Learned Decoder " + str(basic_dict['snr_range'][index]))
                    utils.plot_decoding(datasets[index], rule_classification, basic_dict, 1, "Rule Decoder " + str(basic_dict['snr_range'][index]))
                    utils.plot_decoding(datasets[index], estimate_classification, basic_dict, 1, "Estimated Decoder " + str(basic_dict['snr_range'][index]))
                    utils.plot_decoding(datasets[index], no_learn_classification, basic_dict, 1, "Trivial Decoder " + str(basic_dict['snr_range'][index]))
        all_errors = [total_errors/n_cycles, total_rule_errors/n_cycles, total_estimator_errors/n_cycles, total_no_learning_errors/n_cycles]
        utils.plot_snr_error_rate(all_errors, self.labels, basic_dict)
        return all_errors

    def log_run_info(self, basic_dict):
        file1 = open("log.txt", "w")
        file1.write("Input dimension: " + str(basic_dict['d_x']) + " Output dimension:" + str(basic_dict['d_y']) + "\n")
        file1.write("Number of codewords: " + str(basic_dict['m']) + "\n")
        file1.write("Codebook type: " + basic_dict['codebook_type'] + "\n")
        file1.write("Codewords maximal energy: " + str(basic_dict['codeword_energy']) + "\n")
        file1.write("Number of train samples: " + str(basic_dict['n']) + "\n")
        file1.write("Number of test samples: " + str(basic_dict['n']*basic_dict['test_n_ratio']) + "\n")
        file1.write("Noise type: " + basic_dict['noise_type'] + "\n")
        if basic_dict['df'] is not None:
            file1.write("Degrees of freedom: " + str(basic_dict['df']) + "\n")
        file1.write("Noise energy: " + str(basic_dict['noise_energy']) + "\n")
        file1.write("Training SNR: " + str(basic_dict['train_snr']) + "\n")
        if self.snr_test:
            file1.write("SNR test range:\n")
            file1.write(str(basic_dict['snr_range']) + "\n")
            file1.write("SNR test cycles: " + str(basic_dict['snr_test_cycles']) + "\n")
            file1.write("Number of samples for each SNR: " + str(basic_dict['snr_val_size']) + "\n")
        file1.write("Number of iterations: " + str(basic_dict['iterations']) + "\n")
        file1.write("Batch size: " + str(basic_dict['batch_size']) + "\n")
        file1.write("Error plot iteration gap: " + str(basic_dict['iter_gap']) + "\n")
        if self.lambda_sweep:
            file1.write("Lambda range:\n")
            file1.write("["+str(10**basic_dict['lambda_range'][0])+","+str(10**basic_dict['lambda_range'][1]) + "]\n")
        else:
            file1.write("Lambda: " + str(basic_dict['scale_lambda']) + "\n")
        file1.write("Etas: " + str(basic_dict['etas']) + "\n")
        file1.write("Seed: " + str(basic_dict['seed']) + "\n")
        file1.write("SNR test seed: " + str(basic_dict['snr_seed']) + "\n")
        if basic_dict['code_cov'] is not None:
            file1.write("Random codebook covariance:" + "\n")
            file1.write(str(basic_dict['code_cov']) + "\n")
        if basic_dict['noise_cov'] is not None:
            file1.write("Random noise covariance(s):" + "\n")
            file1.write(str(basic_dict['noise_cov']) + "\n")
        if basic_dict['mix_means'] is not None:
            file1.write("Random noise mean(s):" + "\n")
            file1.write(str(basic_dict['mix_means']) + "\n")
        if basic_dict['mix_dist'] is not None:
            file1.write("Gaussian mixture distribution:" + "\n")
            file1.write(str(basic_dict['mix_dist']) + "\n")
        file1.write("Iteration with minimum test error: " + str(basic_dict['min_test_iter']) + "\n")
        file1.write("Matrix initialization: " + str(basic_dict['init_matrix']) + "\n")
        file1.write("SGD run seed: " + str(basic_dict['batch_seed']) + "\n")
        file1.close()

    def save_data(self, codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict):
        f = open('codebook.npy', 'wb')
        np.save(f, codebook)
        f.close()
        f = open('train_noise_dataset.npy', 'wb')
        np.save(f, train_noise_dataset)
        f.close()
        f = open('test_noise_dataset.npy', 'wb')
        np.save(f, test_noise_dataset)
        f.close()
        outfile = open("basic_dict", 'wb')
        pickle.dump(basic_dict, outfile)
        outfile.close()

    def experiment(self):
        solution = None
        if self.load:
            code_path = os.getcwd()
            saved_dir, basic_dict, codebook, train_noise_dataset, test_noise_dataset, solution = self.load_data()
            os.chdir(code_path)
            utils.make_run_dir(self.load, saved_dir, basic_dict)
            np.random.seed(basic_dict['seed'])
        else:
            basic_dict = self.init_dict()
            utils.make_run_dir(self.load, None, basic_dict)
            np.random.seed(basic_dict['seed'])
            codebook = self.gen_codebook(basic_dict)
            train_noise_dataset, test_noise_dataset = self.gen_noise_datasets(basic_dict)
        rule = self.get_rule(basic_dict)
        train_dataset = self.transform_dataset(codebook, train_noise_dataset, basic_dict, rule)
        test_dataset = self.transform_dataset(codebook, test_noise_dataset, basic_dict, rule)
        basic_dict['estimation_decoder'] = self.get_estimation_decoder(codebook, train_dataset, train_noise_dataset)
        utils.plot_dataset(train_dataset, basic_dict['train_snr'], codebook, basic_dict)
        if not self.load_sol_array:
            solution = self.run_algorithm(basic_dict, codebook, train_dataset, test_dataset)
        elif self.load_errors:
            utils.plot_error_rate(basic_dict['train_errors'], basic_dict['train_rule_error'],
                                  basic_dict['train_estimator_error'], basic_dict['test_errors'],
                                  basic_dict['test_rule_error'], basic_dict['test_estimator_error'], basic_dict['obj_vals'])
        if self.snr_test:
            self.run_snr_test(basic_dict, codebook, solution)
        if self.just_replot_SNR:
            all_errors = [basic_dict['snr_errors'], basic_dict['snr_rule_errors'], basic_dict['snr_estimation_errors'],
                          basic_dict['snr_naive_errors']]
            utils.plot_snr_error_rate(all_errors, self.labels, basic_dict)
        if self.save:
            self.save_data(codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict)
        self.log_run_info(basic_dict)
