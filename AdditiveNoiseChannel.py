from CommChannel import CommChannel
import numpy as np
from numpy import linalg as LA
import utils


class AdditiveNoiseChannel(CommChannel):
    def __init__(self):
        super().__init__(model="MNN")
        self.labels = [r'$S_T$', r'$\Sigma^{-1}$', r'$\hat{\Sigma}^{-1}$', r'$I$']

    def load_sol(self):
        f = open('s_array.npy', 'rb')
        s_array = np.load(f)
        f.close()
        return [s_array]

    def init_dict(self):
        d = 2
        basic_dict = {"d_x": d, "d_y": d, "m": 64, "n": 20, "test_n_ratio": 10, "iterations": 1000,
                      "scale_lambda": 10**-7, "etas": (d+1)*[1/(d+1)], "seed": 25, "codebook_type": "Grid",
                      "codeword_energy": 1, "noise_type": "Mixture", "noise_energy": 0.005, "snr_steps": 10,
                      "snr_seed": 777, "lambda_range": [-7, -5], "batch_size": 5, "model": "MNN", "iter_gap": 1,
                      "snr_val_size": 10000, "snr_test_cycles": 20, "init_matrix": "identity", "batch_seed": 752}
        return basic_dict

    def get_rule(self, basic_dict):
        if len(basic_dict['noise_cov'].shape) == 2:
            precision = LA.inv(basic_dict['noise_cov'])
        else:
            precision = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
            for i in range(len(basic_dict['mix_dist'])):
                precision += basic_dict['mix_dist'][i] * basic_dict['noise_cov'][i]
            precision = LA.inv(precision)
        return precision

    def transform_dataset(self, codebook, noise_dataset, basic_dict, rule=None):
        dataset = np.zeros((basic_dict['m'], noise_dataset.shape[0], basic_dict['d_x']))  # dataset is m x n x d
        for i in range(len(codebook)):
            dataset[i] = noise_dataset + codebook[i]
        return dataset

    def subgradient_alg(self, basic_dict, codebook, dataset, scale_lambda, partition, deltas):
        s_array = []
        obj_vals = np.zeros(basic_dict['iterations'])
        np.random.seed(basic_dict['batch_seed'])
        if basic_dict['init_matrix'] == "identity":
            s = np.eye(basic_dict['d_x'])
        else:
            s = np.zeros((basic_dict['d_x'], basic_dict['d_x']))
        for t in range(1, basic_dict['iterations'] + 1):
            v_t = 0
            for k in range(basic_dict["batch_size"]):
                p_t = np.random.randint(basic_dict['m'] - 1)
                q_t = np.random.randint(p_t + 1, basic_dict['m'])
                x_p_t = np.expand_dims(codebook[p_t], axis=1)
                x_q_t = np.expand_dims(codebook[q_t], axis=1)
                z_t = np.random.randint(basic_dict['n'])
                which_word = np.random.randint(2)
                y_t = np.expand_dims(dataset[p_t, z_t] if which_word == 0 else dataset[q_t, z_t], axis=1)
                delta_p_q_t = np.expand_dims(deltas[utils.double_to_single_index(p_t, q_t, basic_dict['m'])], axis=1)
                a_t = ((-1) ** which_word) * (y_t - 0.5 * (x_p_t + x_q_t))
                if a_t.T @ s @ delta_p_q_t < 1:
                    v_t += delta_p_q_t @ a_t.T + a_t @ delta_p_q_t.T - np.diag(np.diag(a_t @ delta_p_q_t.T))
                obj_vals[t-1] += (1/basic_dict["batch_size"]) * max(0, 1-a_t.T@s@delta_p_q_t)
            grad_t = 0
            for i, p_i in enumerate(partition):
                delta_p_q_star = np.expand_dims(p_i[np.argmax(LA.norm(np.dot(s, p_i.T), axis=0) ** 2)], axis=1)
                grad_t += basic_dict['etas'][i] * (
                            s @ delta_p_q_star @ delta_p_q_star.T + delta_p_q_star @ delta_p_q_star.T @ s - np.diag(np.diag(s @ delta_p_q_star @ delta_p_q_star.T)))
                obj_vals[t-1] += scale_lambda * basic_dict['etas'][i] * (LA.norm(s@delta_p_q_star) ** 2)
            grad_t = scale_lambda * grad_t - v_t / basic_dict["batch_size"]
            s -= (1 / (scale_lambda * t)) * grad_t
            s = utils.get_near_psd(s)
            s_array.append(s)
        return [s_array], obj_vals

    def get_true_classification(self, basic_dict, n_samples):
        return np.array([i for i in range(basic_dict['m']) for j in range(n_samples)])

    def decode(self, codebook, dataset, decoder):
        m, n, d = dataset.shape
        examples = dataset.reshape(m * n, d)
        examples_minus_codewords = np.repeat(examples, m, axis=0) - np.tile(codebook, (m * n, 1))
        a = np.einsum('ij,ji->i', examples_minus_codewords @ decoder, examples_minus_codewords.T)
        b = np.reshape(a, (n * m, m))
        classification = np.argmin(b, axis=1)
        return classification

    def rule_decode(self, codebook, dataset, precision):
        return self.decode(codebook, dataset, precision)

    def no_learning_decode(self, codebook, dataset):
        return self.decode(codebook, dataset, np.eye(dataset.shape[2]))

    def get_estimation_decoder(self, codebook, dataset, noise_dataset):
        n, d = noise_dataset.shape
        return LA.pinv((noise_dataset.T @ noise_dataset) / (n-1))

    def estimator_decode(self, codebook, dataset, estimator):
        return self.decode(codebook, dataset, estimator)

    def save_data(self, codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict):
        super().save_data(codebook, train_noise_dataset, test_noise_dataset, solution, basic_dict)
        f = open('s_array.npy', 'wb')
        np.save(f, solution[0])
        f.close()


if __name__ == '__main__':
    channel = AdditiveNoiseChannel()
    channel.experiment()
