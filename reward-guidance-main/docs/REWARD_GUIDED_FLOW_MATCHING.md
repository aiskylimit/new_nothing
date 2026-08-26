# Reward-Guided Flow Matching: implemented specification

This repository maximizes a reward with the normalized endpoint target

\[
p_\beta(x\mid c)=\frac{p_{\mathrm{data}}(x\mid c)
\exp(\beta r(x,c))}{Z_\beta(c)}, \qquad \beta\ge 0.
\]

The normalization constant is required for a probability distribution, but it
does not need to be known when finite data are resampled with normalized
log-weights.

## Correct Taylor signs

Let the base local component be `N(mu, sigma^2 I)`, let
`g = grad r(mu)`, and let `H = Hessian r(mu)`. Expanding the positive reward
tilt `exp(+beta r)` gives

\[
\widetilde p_{\text{first}} =
\mathcal N(\mu + \beta\sigma^2g,\,\sigma^2I)
\]

and

\[
\Sigma=(\sigma^{-2}I-\beta H)^{-1},\qquad
\widetilde p_{\text{second}}=
\mathcal N(\mu+\beta\Sigma g,\,\Sigma).
\]

The alternative signs `mu - beta Sigma g` and
`(sigma^-2 I + beta H)^-1` correspond to `exp(-beta r)`, i.e. treating the
reward as a cost.

## Training the tilted velocity field

Normalizing every conditional component
`N(x | mu_t(x0), sigma_t^2 I) exp(beta_t r(x))` independently does **not**
produce the tilted endpoint: at zero variance, the component normalization
cancels the reward factor. Keeping the components unnormalized makes their
mixture weights time-dependent, for which an ordinary conditional
flow-matching target is insufficient.

The implemented training path therefore samples the desired endpoint first:

1. Compute rewards for the empirical data pool.
2. Normalize `log w_i = beta * r_i` with a log-sum-exp/softmax.
3. Resample endpoints according to `w_i`.
4. Apply standard conditional flow matching between Gaussian noise and those
   tilted endpoints.

This has the exact empirical terminal marginal `p_beta`; no unknown partition
function or heuristic gradient normalization is involved. See
`checkerboard/train.py --target-distribution reward_tilted`.

The first/second-order formulas remain useful for inference-time conditional
guidance and are shared through `reward_guidance_math.py`.

## Low-dimensional and FLUX implementations

- Checkerboard uses the full analytic reward Hessian. Positive curvature is
  retained and capped only when the local Gaussian precision would fall below
  `--precision-floor`.
- FLUX cannot form a full image/latent Hessian on an H200. It uses the explicit
  approximation `J_f^T H_head J_f`, selects the largest-magnitude curvature
  directions, and solves the low-rank system with Woodbury. Both negative and
  positive curvature are retained; positive curvature is damped only as much
  as required for a positive-definite precision.
- `--beta` is the exponential-tilt coefficient. A fixed
  `--gradient-norm-scale` is retained only for comparisons with the original
  guidance experiments; `--gradient-norm-scale 0` uses the beta-faithful raw
  gradient.

The FLUX method is intentionally described as a **feature-space Gauss-Newton
second-order approximation**, not a full input-space Hessian.

## Validation

Run the local mathematical and runtime checks:

```bash
python -m unittest discover -s tests -v
python flux/smoke_test_second_order.py
```

After training the reward-tilted checkerboard checkpoint, compare it with the
exact rejection-sampled target:

```bash
cd checkerboard
python train.py --num-steps 500000
python train.py \
  --target-distribution reward_tilted --beta 10 \
  --init-checkpoint results/velocity_net.pt \
  --num-steps 100000 --output-dir results/reward_tilted_beta10
python evaluate_reward_tilted.py \
  --model-dir results/reward_tilted_beta10 --beta 10
```

The evaluator records reward error, checkerboard valid mass, histogram
Jensen-Shannon divergence, and sliced Wasserstein-2 distance. Reward alone is
not accepted as evidence that the learned distribution is correct.

## H200 compatibility

The existing FLUX loading path, fused Flow Map LoRA, math-only SDPA,
gradient-checkpointing, BF16 transformer/VAE execution, FP64 eigendecomposition,
and H200 preflight are preserved. The new precision stabilization operates only
on the selected low-rank matrix (rank 16 by default), so it does not create a
full latent Hessian or covariance.
