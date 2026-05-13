# Contributing

Thanks for your interest in SuperKittens. See [`dev_env.md`](./dev_env.md) for environment setup, repository layout, and the "adding a new model" template.

## Quick rules

- **One PR per logical change.** Don't bundle a refactor with a feature.
- **Validate against HuggingFace `transformers`.** Every model port must reach `argmax(SK_logits) == argmax(HF_logits)` with `rel_err < 0.1` on a real prompt before being called done.
- **Default to no comments.** Identifiers should be self-explanatory; comments explain WHY, never WHAT.
- **Don't touch other model families' code** when porting a new one. Only `kernels/` is shared.
- **No emojis in code or files** unless explicitly requested.
- **Clean build is mandatory.** `./build.sh` must exit non-zero on any compile failure — no silent skips.

## Pull requests

- Open the PR against `main` from a feature branch named `dev-<family>` or `dev-<topic>`.
- Description should state: what changed, how it was validated (argmax + rel_err + tok/s), and any known follow-ups.
- Direct pushes to `main` are blocked. All changes land via PR review.

## Bug reports

Open a GitHub issue with:
- Model + prompt that reproduces.
- `rel_err` per layer (use the `layer_diff.py` harness in `temp/<family>_validate/`).
- First divergent step, if known.
