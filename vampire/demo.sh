set -eux

mkdir -p pipe_main/_output_demo/02-0249_TCRB.4000/02-0249_TCRB.4000.head/
python3 preprocess_adaptive.py --sample 1200 pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 pipe_main/_output_demo/02-0249_TCRB.4000/train.processed.csv
python3 util.py split --train-size 1000 pipe_main/_output_demo/02-0249_TCRB.4000/train.processed.csv pipe_main/_output_demo/02-0249_TCRB.4000/training.csv pipe_main/_output_demo/02-0249_TCRB.4000/validation.csv
gene-name-conversion adaptive2olga pipe_main/_output_demo/02-0249_TCRB.4000/training.csv pipe_main/_output_demo/02-0249_TCRB.4000/training.olga.tsv
python3 preprocess_adaptive.py --sample 100 pipe_main/sample_data/02-0249_TCRB.4000.tsv.bz2 pipe_main/_output_demo/02-0249_TCRB.4000/02-0249_TCRB.4000.head.csv
echo '{"model": "basic", "latent_dim": 20, "dense_nodes": 75, "aa_embedding_dim": 21, "v_gene_embedding_dim": 30, "j_gene_embedding_dim": 13, "beta": 0.75, "max_cdr3_len": 30, "n_aas": 21, "n_v_genes": 59, "n_j_genes": 13, "stopping_monitor": "val_loss", "batch_size": 100, "pretrains": 2, "warmup_period": 3, "epochs": 10, "patience": 20}' > pipe_main/_output_demo/02-0249_TCRB.4000/model_params.json
python3 execute.py --clusters='' --script-prefix=train 'pipe_main/_output_demo/02-0249_TCRB.4000/model_params.json pipe_main/_output_demo/02-0249_TCRB.4000/training.csv' 'pipe_main/_output_demo/02-0249_TCRB.4000/best_weights.h5 pipe_main/_output_demo/02-0249_TCRB.4000/diagnostics.csv' 'tcr-vae train {sources} {targets}'
python3 execute.py --clusters='' --script-prefix=pvae 'pipe_main/_output_demo/02-0249_TCRB.4000/model_params.json pipe_main/_output_demo/02-0249_TCRB.4000/best_weights.h5 pipe_main/_output_demo/02-0249_TCRB.4000/02-0249_TCRB.4000.head.csv' 'pipe_main/_output_demo/02-0249_TCRB.4000/02-0249_TCRB.4000.head/test.pvae.csv' 'tcr-vae pvae {sources} {targets}'
tcr-vae generate --nseqs 100 pipe_main/_output_demo/02-0249_TCRB.4000/model_params.json pipe_main/_output_demo/02-0249_TCRB.4000/best_weights.h5 pipe_main/_output_demo/02-0249_TCRB.4000/vae-generated.csv
