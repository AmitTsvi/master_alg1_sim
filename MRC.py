import numpy as np
from scipy.stats import norm
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pickle


def plot_kl(sigma_s_l, SNR_l, kl_l):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SNR_l, kl_l, color='blue', marker='s', linewidth=2, label="KL")
    ax.plot(SNR_l, sigma_s_l, color='green', marker='s', linewidth=2, label=r'$\sigma^2$')
    plt.legend()
    plt.grid()
    plt.xlabel("SNR[dB]")
    plt.savefig('Result')
    plt.show()
    plt.close()


def KL(sigma_s, mu, sigma_prior, N):
    return (N/2)*np.log2(sigma_prior/sigma_s)-(N/2)+(N/2)*(sigma_s/sigma_prior)+LA.norm(mu)/(2*sigma_prior)


def SGD(codebook, input_samples, output_samples, N):
    mu = np.zeros(N)
    sigma_s = 1
    sigma_prior = 10000
    eta = 0.01
    iterations = 10000
    batch_size = 10
    n_samples = output_samples.shape[1]

    for it in range(iterations):
        samples_idx = np.random.randint(n_samples, size=batch_size)
        for s_idx in samples_idx:
            x_i = input_samples[:, s_idx]
            y_i = output_samples[:, s_idx]
            ele1_mu = np.sum([(y_i*(x_i-x) / (LA.norm(y_i)*np.abs(x_i-x)*np.sqrt(sigma_s)))
                          * norm.pdf((-np.dot(mu, y_i)*(x_i-x)+0.5*(x**2-x_i**2))
                                     / (LA.norm(y_i)*np.abs(x_i-x)*np.sqrt(sigma_s))) for x in codebook if x != x_i])
            mu = mu + eta*ele1_mu - eta/sigma_prior*mu
            ele1_sigma = np.sum([(np.dot(mu, y_i)*(x_i-x)+0.5*(x**2-x_i**2))
                                     / (2*LA.norm(y_i)*np.abs(x_i-x)*sigma_s**1.5)
                          * norm.pdf((-np.dot(mu, y_i)*(x_i-x)+0.5*(x**2-x_i**2))
                                     / (LA.norm(y_i)*np.abs(x_i-x)*np.sqrt(sigma_s))) for x in codebook if x != x_i])
            sigma_s = sigma_s - eta * ele1_sigma + eta*N / (2*sigma_s) - eta*N / (2*sigma_prior)
    return mu, sigma_s, KL(sigma_s, mu, sigma_prior, N)


def main():
    d = 1
    N = 4
    m = 7
    n = 1000
    gains = np.random.randint(5, size=(N, 1))
    codebook = list(np.arange(-int(m/2), m-int(m/2)))
    code_energy = np.mean(np.sum(np.power(codebook, 2)))

    sigma_noise_range = [2**i for i in range(-20, 20)]
    mu_l = []
    sigma_s_l = []
    kl_l = []
    SNR_l = []
    for sigma_noise in sigma_noise_range:
        noise_energy = sigma_noise * N
        SNR = 10 * np.log10(code_energy / noise_energy)
        input_samples = np.expand_dims(np.repeat(codebook, n), 0)
        noise_samples = np.random.normal(0, sigma_noise, (N, n * m))
        output_samples = input_samples + gains * noise_samples
        mu, sigma_s, kl = SGD(codebook, input_samples, output_samples, N)
        mu_l.append(mu)
        sigma_s_l.append(sigma_s)
        kl_l.append(kl)
        SNR_l.append(SNR)
    results = {"SNR": SNR_l, "KL": kl_l, "Sigma": sigma_s_l, "Mu": mu_l, "Gains": gains}
    outfile = open("results", 'wb')
    pickle.dump(results, outfile)
    outfile.close()
    plot_kl(sigma_s_l, SNR_l, kl_l)


if __name__ == '__main__':
    main()
