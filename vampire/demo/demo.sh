set -eux

mkdir -p _output_demo

# Preprocess an Adaptive sequence file.
python3 ../preprocess_adaptive.py --sample 1000 ../pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 _output_demo/train.processed.csv
python3 ../preprocess_adaptive.py --sample 100 ../pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 _output_demo/100.csv

# Train our VAE using the supplied model parameters and save the best weights.
tcr-vae train model_params.json _output_demo/train.processed.csv _output_demo/best_weights.h5 _output_demo/diagnostics.csv

# Calculate some P_VAE values.
tcr-vae pvae model_params.json _output_demo/best_weights.h5 _output_demo/100.csv _output_demo/100.pvae.csv

# Generate some sequences from our VAE. Note that these sequences are going to
# look silly and repetitive because this is a very badly trained VAE: not
# enough sequence data and insufficient training time.
tcr-vae generate --nseqs 100 model_params.json _output_demo/best_weights.h5 _output_demo/vae-generated.csv
