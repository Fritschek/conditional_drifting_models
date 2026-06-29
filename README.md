# Conditional Drifting Models for Learned Channel Simulation

Code for the paper "Conditional Drifting Models for Learned Channel Simulation".

The repository implements the generator-level channel simulators used in the
paper:

- direct kernel drifting on channel outputs;
- joint Sinkhorn drifting on `(x,y)`;
- condition-wise Sinkhorn drifting on repeated samples of `p(y | x)`.

The included channels are AWGN, Rayleigh fading, SSPA/Rapp nonlinearity, and a
compact TDL fading channel. Paper defaults are collected in
`conditional_drifting/config.py`.

## Run a Channel Surrogate

```bash
python scripts/run_drifting_benchmark.py \
  --channel Rayleigh \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 30 \
  --dataset-size 1000000 \
  --eval-size 200000 \
  --batch-size 5000 \
  --out results/rayleigh_conditionwise_seed7.json \
  --checkpoint results/rayleigh_conditionwise_seed7.pt
```

Use `--variant direct`, `--variant joint_sinkhorn`, or
`--variant conditionwise_sinkhorn` to select the generator. For multi-seed
experiments, run the same command over the desired seeds and channels.

## Quick Matrix Check

The following command exercises all public channel and generator paths:

```bash
python scripts/run_quick_model_checks.py \
  --device cuda \
  --seed 7 \
  --epochs 2 \
  --dataset-size 2048 \
  --eval-size 2048 \
  --batch-size 128 \
  --out results/quick_model_checks.json
```

## Paper-Scale Generator Runs

The larger generator-level settings use the same command-line knobs. For AWGN
and Rayleigh:

```bash
python scripts/run_drifting_benchmark.py \
  --channel AWGN \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 30 \
  --dataset-size 10000000 \
  --eval-size 10000000 \
  --batch-size 5000 \
  --out results/awgn_conditionwise_paper_seed7.json
```

For SSPA:

```bash
python scripts/run_drifting_benchmark.py \
  --channel SSPA \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 160 \
  --dataset-size 10000000 \
  --eval-size 10000000 \
  --batch-size 4096 \
  --out results/sspa_conditionwise_paper_seed7.json
```

For TDL:

```bash
python scripts/run_drifting_benchmark.py \
  --channel TDL \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 60 \
  --dataset-size 120000 \
  --eval-size 100000 \
  --batch-size 512 \
  --out results/tdl_conditionwise_paper_seed7.json
```

## AWGN Reference Check

With the paper AWGN preset and seed 7, condition-wise Sinkhorn direct-output SWD
improves as the training budget is increased:

- 120k training samples, 60 epochs, batch size 512: SWD `1.55e-2`;
- 1M training samples, 30 epochs, batch size 5000: SWD `9.91e-3`;
- 10M training samples, 30 epochs, batch size 5000: SWD `7.95e-3`.

The script `scripts/run_awgn_ser_ber_check.py` trains a small symbolic
autoencoder through the AWGN surrogate and evaluates it on the analytic AWGN
channel. With the 1M-sample surrogate setting above and a 10-epoch autoencoder
check, seed 7 gave analytic-channel evaluation SER/BER `9.22e-3 / 5.19e-3`
after analytic-channel training and `9.36e-3 / 5.11e-3` after surrogate-channel
training.

## License

MIT License. See `LICENSE`.
