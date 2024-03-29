from CommChannel import CommChannel
import numpy as np
import utils
from scipy.spatial import distance
from numpy import linalg as LA


class NonLinearChannel(CommChannel):
    def __init__(self):
        super().__init__(model="LTNN")
        self.channel_trans = None
        self.labels = [r'$H_T$', r'$f(\cdot)$', r'$\mu_x$']

    def load_sol(self):
        f = open('s_array.npy', 'rb')
        s_array = np.load(f)
        f.close()
        f = open('h_array.npy', 'rb')
        h_array = np.load(f)
        f.close()
        return [h_array, s_array]

    def init_dict(self):
        d_x = 2
        d_y = 2
        basic_dict = {"d_x": d_x, "d_y": d_y, "m": 8, "n": 800, "test_n_ratio": 1, "iterations": 1400,
                      "scale_lambda": (10**-3, 10**-4),  "etas": (d_x+1)*[1/(d_x+1)], "seed": 5, "codebook_type": "TwoCircles",
                      "codeword_energy": 1, "noise_type": "WhiteGaussian", "noise_energy": 0.5, "snr_steps": 10,
                      "snr_seed": 6, "lambda_range": [-3.5, -2.5], "batch_size": 10, "model": "LTNN", "iter_gap": 1,
                      "snr_val_size": 10000, "snr_test_cycles": 20, "init_matrix": "zeros", "batch_seed": 5,
                       "trans_type": "Linear_Invertible", "max_eigenvalue": 2, "min_eigenvalue": 1, "with_s": True}
        return basic_dict

    def get_rule(self, basic_dict):
        if self.channel_trans is None:
            if self.load:
                self.channel_trans = utils.rebuild_trans_from_kernel(basic_dict['trans_kernel'], basic_dict['trans_type'])
            else:
                self.channel_trans = self.gen_transformation(basic_dict)
        return self.channel_trans

    def gen_transformation(self, basic_dict):
        channel_trans, trans_kernel = utils.gen_transformation(basic_dict['d_x'], basic_dict['d_y'],
                                                               basic_dict['trans_type'],
                                                               basic_dict['max_eigenvalue'],
                                                               basic_dict['min_eigenvalue'])
        basic_dict['trans_kernel'] = trans_kernel
        return channel_trans

    def transform_dataset(self, codebook, noise_dataset, basic_dict, trans=None):
        transformed_codewords = np.array([trans(x) for x in codebook])
        dup_trans_codewords = np.repeat(transformed_codewords, int(noise_dataset.shape[0] / basic_dict['m']), axis=0)  # n x d_y
        return dup_trans_codewords + noise_dataset

    def subgradient_alg(self, basic_dict, codebook, dataset, scale_lambda, partition, deltas):
        if type(scale_lambda) is not tuple:
            scale_lambda = (scale_lambda, scale_lambda)
        s_array = []
        h_array = []
        obj_vals = np.zeros(basic_dict['iterations'])
        np.random.seed(basic_dict['batch_seed'])
        if basic_dict['init_matrix'] == "identity":
            s = np.eye(basic_dict['d_x'])
            h = np.eye(basic_dict['d_y'], basic_dict['d_x'])
        else:
            s = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
            h = np.zeros((basic_dict['d_y'], basic_dict['d_x']))
        iter_size = basic_dict["batch_size"] * (basic_dict['m'] - 1)
        print("Starting algorithm run with " + str(scale_lambda))
        for t in range(1, basic_dict['iterations'] + 1):
            v_h_t = 0
            v_s_t = 0
            batch_indices = np.random.randint(basic_dict['n'], size=basic_dict["batch_size"])
            for k in range(basic_dict["batch_size"]):
                y_t = np.expand_dims(dataset[batch_indices[k]], axis=1)
                x_j = np.expand_dims(codebook[int(batch_indices[k] // (basic_dict['n'] / basic_dict['m']))], axis=1)
                for x_tag in codebook:
                    x_tag_e = np.expand_dims(x_tag, axis=1)
                    words_plus = x_j + x_tag_e
                    words_minus = x_j - x_tag_e
                    if not np.array_equal(x_tag_e, x_j):
                        if basic_dict["with_s"]:
                            if y_t.T @ h @ words_minus - 0.5 * words_plus.T @ s @ words_minus < 1:
                                v_h_t += y_t @ words_minus.T / iter_size
                                v_s_t += (words_plus @ words_minus.T + words_minus @ words_plus.T - np.diag(np.diag(words_plus @ words_minus.T))) / iter_size
                            obj_vals[t - 1] += max(0, 1 - (
                                        y_t.T @ h @ words_minus - 0.5 * words_plus.T @ s @ words_minus)) / iter_size
                        else:
                            if y_t.T @ h @ words_minus - 0.5 * words_plus.T @ h.T @ h @ words_minus < 1:
                                v_h_t += (y_t @ words_minus.T - 0.5 * h @ (words_plus @ words_minus.T + words_minus @ words_plus.T)) / iter_size
                            obj_vals[t - 1] += max(0, 1 - (
                                        y_t.T @ h @ words_minus - 0.5 * words_plus.T @ h.T @ h @ words_minus)) / iter_size
            grad_h_t = 0
            grad_s_t = 0
            for i, p_i in enumerate(partition):
                delta_h = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(h, p_i.T), axis=0) ** 2)], axis=1)
                grad_h_t += basic_dict['etas'][i] * 2 * (h @ delta_h @ delta_h.T)
                delta_s = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
                grad_s_t += basic_dict['etas'][i] * 2 * (s @ delta_s @ delta_s.T + delta_s @ delta_s.T @ s - np.diag(np.diag(s @ delta_s @ delta_s.T)))
                obj_vals[t-1] += scale_lambda[0] * basic_dict['etas'][i] * (LA.norm(h @ delta_h) ** 2)
                if basic_dict["with_s"]:
                    obj_vals[t-1] += scale_lambda[1] * basic_dict['etas'][i] * (LA.norm(s @ delta_s) ** 2)
            grad_h_t = scale_lambda[0] * grad_h_t - v_h_t
            grad_s_t = scale_lambda[1] * grad_s_t + 0.5 * v_s_t
            if scale_lambda[0] == 0 or scale_lambda[1] == 0:
                h -= (1 / t) * grad_h_t
                s -= (1 / t) * grad_s_t
            else:
                h -= (1 / (scale_lambda[0] * t)) * grad_h_t
                s -= (1 / (scale_lambda[1] * t)) * grad_s_t
            if basic_dict["with_s"]:
                h, s = utils.projection(h, s)
            h_array.append(h)
            s_array.append(s)
        return [h_array, s_array], obj_vals

    def get_true_classification(self, basic_dict, n_samples):
        return np.repeat(np.array([i for i in range(basic_dict['m'])]), int(n_samples / basic_dict['m']), axis=0)

    def decode(self, codebook, dataset, decoder):
        n = dataset.shape[0]
        m = codebook.shape[0]
        transformed_codebook = (decoder @ codebook.T).T
        examples_minus_codewords = np.repeat(dataset, m, axis=0) - np.tile(transformed_codebook, (n, 1))
        a = np.einsum('ij,ji->i', examples_minus_codewords, examples_minus_codewords.T)
        b = np.reshape(a, (n, m))
        classification = np.argmin(b, axis=1)
        return classification

    def rule_decode(self, codebook, dataset, trans):
        inv_trans = lambda y, cb: np.argmin([distance.euclidean(trans(c), y) for c in cb])
        return np.apply_along_axis(inv_trans, 1, dataset, codebook)

    def no_learning_decode(self, codebook, dataset):
        return None

    def get_estimation_decoder(self, codebook, dataset, noise_dataset):
        n = dataset.shape[0]
        m = codebook.shape[0]
        sample_per_word = int(n / m)
        return np.array([np.mean(dataset[sample_per_word * i:sample_per_word * (i + 1) - 1], axis=0) for i in range(m)])

    def estimator_decode(self, codebook, dataset, estimator):
        n = dataset.shape[0]
        m = codebook.shape[0]
        examples_minus_codewords = np.repeat(dataset, m, axis=0) - np.tile(estimator, (n, 1))
        a = np.einsum('ij,ji->i', examples_minus_codewords, examples_minus_codewords.T)
        b = np.reshape(a, (n, m))
        return np.argmin(b, axis=1)

    def save_data(self, codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict):
        super().save_data(codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict)
        f = open('h_array.npy', 'wb')
        np.save(f, solution[0])
        f.close()
        f = open('s_array.npy', 'wb')
        np.save(f, solution[1])
        f.close()

    def log_run_info(self, basic_dict):
        super().log_run_info(basic_dict)
        file1 = open("log.txt", "a")
        file1.write("Transformation type: " + basic_dict["trans_type"] + "\n")
        file1.write("Transformation singular values between " + str(basic_dict["min_eigenvalue"]) + "and " + str(basic_dict["max_eigenvalue"]) + "\n")
        file1.write("Transformation Kernels:" + "\n")
        file1.write(str(basic_dict['trans_kernel']) + "\n")
        file1.write("Trained only with gradients w.r.t. H: " + str(not basic_dict["with_s"]))
        file1.close()


if __name__ == '__main__':
    channel = NonLinearChannel()
    channel.experiment()
