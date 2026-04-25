# Phase-2 Submission Pack (OpenEnv Hackathon India 2026)

Use this as the final pre-submit checklist.

## 1) Required Deliverables

- [ ] OpenEnv environment uses latest release (`openenv-core[core]>=0.2.3`).
- [ ] Environment repo pushed and runnable on Hugging Face Space.
- [ ] Minimal TRL/GRPO training path in Colab:
  - [ ] [`notebooks/minimal_trl_grpo_colab.ipynb`](./notebooks/minimal_trl_grpo_colab.ipynb)
  - [ ] [`notebooks/logiflow_grpo_colab.ipynb`](./notebooks/logiflow_grpo_colab.ipynb) (fuller version)
  - [ ] [`notebooks/colab_space_grpo_unsloth.ipynb`](./notebooks/colab_space_grpo_unsloth.ipynb) (single-file Space-connected Unsloth + TRL GRPO workflow)
- [ ] Training evidence included (from a real run):
  - [ ] Reward curve PNG
  - [ ] Before/after CSV or JSON
  - [ ] Short summary metrics
- [ ] One mini-blog (HF) or one short video (`<2 min`) published.
- [ ] README contains all links judges need.

## 2) Commands To Reproduce

```bash
pip install -e ".[train]"
python test_engine.py
python train_and_evaluate.py
python train_grpo.py --model-name Qwen/Qwen2.5-0.5B-Instruct --max-steps 140
```

Expected GRPO artifacts:

- `outputs/logiflow-grpo-script/artifacts/reward_curve.png`
- `outputs/logiflow-grpo-script/artifacts/evaluation_before.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_after.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_summary.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_summary.json`
- `outputs/logiflow-grpo-script/artifacts/improvement.json`

## 3) Links To Fill Before Final Submit

- HF Space URL: `<paste-your-space-url>`
- Colab URL (minimal): `<paste-colab-url>`
- Colab URL (full): `<paste-colab-url>`
- Mini-blog or short video URL: `<paste-url>`
- Optional W&B run URL: `<paste-url>`

## 4) Fast Sanity Criteria (Judge Lens)

- Innovation: show dynamic pressure + cascading disruption + priority demand.
- Story: explain `problem -> environment -> reward -> learned behavior`.
- Improvement: include before/after with numeric deltas (not only screenshots).
- Reward pipeline: show anti-hacking checks and reward breakdown fields.
