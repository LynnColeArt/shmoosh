# QJL Research Notes

## Sources Checked

- TurboQuant: <https://arxiv.org/abs/2504.19874>
- QJL: <https://arxiv.org/abs/2406.03482>
- AAAI QJL paper PDF: <https://ojs.aaai.org/index.php/AAAI/article/download/34773/36928>

## Relevant Takeaways

TurboQuant explicitly separates two objectives:

- MSE reconstruction quality;
- unbiased inner-product estimation.

The paper's key point for our work is that MSE-optimal quantizers can be biased for inner products. TurboQuant therefore applies MSE quantization first, then QJL on the residual to estimate the missing inner-product term.

QJL itself is an asymmetric estimator: one vector gets a sign-bit JL sketch, while the query side uses the corresponding unquantized JL projection. The QJL paper states that this gives an unbiased inner-product estimator with small distortion and eliminates the usual block scale/zero-point overhead.

The variance story explains our empirical threshold:

- QJL correction is unbiased in expectation.
- The correction is still noisy for finite sketch size.
- Attention softmax is sensitive to that noise.
- In the Underpaint SDXL fixture, 16 and 32 residual signs made logits worse, 64 was borderline, and 128 was the first stable setting.

## Design Implication

Turbo-D should not treat QJL as a tiny optional add-on. For attention logits, QJL needs enough sketch width to reduce variance below softmax sensitivity. The current prototype should use QJL-128 as the baseline and investigate better scaling, orthogonalized projections, or calibrated shrinkage before trying to reduce sketch width.

## Open Research Questions

1. Would orthogonal or Hadamard-style sketch rows reduce variance enough for QJL-64?
2. Should the residual correction be shrunk before softmax, trading some bias for lower variance?
3. Does diffusion tolerate correction noise differently across timesteps?
4. Does K-only compression remain best once we run actual image generation instead of captured-tensor output metrics?
