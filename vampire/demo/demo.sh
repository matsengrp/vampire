set -eux

mkdir -p _output_demo
python3 ../preprocess_adaptive.py --sample 1200 ../pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 _output_demo/train.processed.csv
python3 ../util.py split --train-size 1000 _output_demo/train.processed.csv _output_demo/training.csv _output_demo/validation.csv
python3 ../preprocess_adaptive.py --sample 100 ../pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 _output_demo/02-0249_TCRB.4000.head.csv
tcr-vae train model_params.json _output_demo/training.csv _output_demo/best_weights.h5 _output_demo/diagnostics.csv
tcr-vae pvae model_params.json _output_demo/best_weights.h5 _output_demo/02-0249_TCRB.4000.head.csv _output_demo/test.pvae.csv
tcr-vae generate --nseqs 100 model_params.json _output_demo/best_weights.h5 _output_demo/vae-generated.csv
