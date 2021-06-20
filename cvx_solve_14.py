import numpy as np
import utils
import cvxpy as cp
import matplotlib.pyplot as plt


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
    deltas = utils.delta_array(L, d, m, codebook)

    A = np.zeros((2*n*L,d))
    row_index = 0
    for i in range(m):  # word p
        for j in range(i+1, m):  # word q
            for k in range(n):  # sample i
                for which_word in range(2):  # which word was transmitted
                    y_t = (1-which_word)*dataset[i, k]+which_word*dataset[j, k]
                    A[row_index] = ((-1)**which_word)*(y_t-0.5*(codebook[i]+codebook[j]))
                    row_index += 1

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    eta = 5
    S = cp.Variable((d, d), PSD=True)
    obj = cp.Minimize(cp.sum([cp.maximum(0, 1-A[r]@S@deltas[int(r/(2*n))]) for r in range(len(A))])
                      +eta*cp.max(cp.power(cp.norm(S@deltas.T, axis=0), 2)))
    constraints = []
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # Print result.
    print("The optimal value is", prob.value)
    print("A solution S is")
    print(S.value)

    classification = utils.decode(codebook, dataset, m, n, d, S.value)
    utils.plot_decoding(dataset, classification, m, n, d, 11, fig)


if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    fig = plt.figure()
    main()