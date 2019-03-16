See `tcr_vae.py` for a description of what is expected of these files.

We use a `_l` variable name convention for layer instantiations before they accept input.

For the KL loss we use the formula

    -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

see [cross validated](https://stats.stackexchange.com/a/7449/139272) for the derivation.
