import numpy as np
import scypy.stats as stats

# All of our above encoding/decoding was based on single point estimates
# i.e. taking a sequence and passing it through the encoder to generate
# the sampling parameters for the latent space and then take one sample
# from this latent space and decode it using the decoder.
# Instead of just decoding one sample from the latent space we could
# decode multiple and do averaging to get a better estimate.

# Let's investigate this by computing unnormalized sequence probabilities
# and find out how many samples are needed for convergence.


def compute_log_probability(one_hot_seq, pwm):
    prod_mat = np.matmul(one_hot_seq, pwm.T)
    log_prod_diag = np.log(prod_mat.diagonal())
    sum_diag = np.sum(log_prod_diag)
    return (sum_diag)


def compute_log_probability_with_importance_sampling(x, x_prob_z, z_mean, z_sd,
                                                     z_sample):
    '''
    x is the input one hot encoded sequence
    x_prob_z is the VAE predicted probability of each position in x
    z_mean is the encoder output
    z_sd is the encoder output
    z_sample is a sample from the posterior of the latent space q(z|x)
    '''
    # This gets us log of p(x|z):
    prod_mat = np.matmul(x, x_prob_z.T)
    log_prod_diag = np.log(prod_mat.diagonal())
    log_p_x_given_z = np.sum(log_prod_diag)  # p(x|z)

    # Then find the importance weight.
    log_p_z = sum(stats.norm.logpdf(z_sample, 0, 1))  # p(z)
    log_q_z_given_x = sum(stats.norm.logpdf(z_sample, z_mean, z_sd))  # q(z|x)
    # Importance weight: p(z)/q(z|x)
    log_imp_weight = log_p_z - log_q_z_given_x
    return (log_p_x_given_z + log_imp_weight)


def run_importance_sampling(x_test, encoder, decoder, mean_log_prob):
    # Repeat with importance sampling:
    Nsamples = 10000
    chunk = 100  # This is slow so we decrease our chunk size
    x = x_test[0:chunk]
    log_p_list = np.zeros((100, Nsamples))
    log_mean_p = np.zeros(
        (100,
         Nsamples))  # This will be a running mean of the log probabilities
    for i in range(Nsamples):
        # Encode the sequences into latent space:
        z_mean, z_log_var = encoder.predict(x)
        z_sd = np.sqrt(np.exp(z_log_var))
        # Sample from the posterior q(z|x)
        z_sample = stats.norm.rvs(z_mean, z_sd)
        x_prob_z = decoder.predict(
            z_sample)  # Position-wise probabilities of x
        for j in range(chunk):
            logp = compute_log_probability_with_importance_sampling(
                x[j], x_prob_z[j], z_mean[j], z_sd[j], z_sample[j])
            log_p_list[j][i] = logp
            log_mean_p[j][i] = mean_log_prob(log_p_list[j][:(i + 1)])
