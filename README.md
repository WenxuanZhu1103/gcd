# Gold-Conditioned Distillation (GCD): training code

A compact implementation of the training methods from *Self-Distillation for Supervised
Fine-Tuning*: SFT, SDFT, and GCD on Tooluse and LiveNews, with Qwen2.5-3B-Instruct and
Qwen3-4B-Base.

GCD trains a student to match a teacher that sees the gold response in its prompt. Both models
read the same gold response token by token (no sampling), and the student minimizes the forward
KL divergence to the teacher's next-token distribution. The teacher is an exponential moving
average (EMA) of the student.

## Files

| File | Contents |
|---|---|
| `data.py` | Data loading, gold-response formatting, student and teacher prompts |
| `train.py` | One training loop for `--method sft`, `gcd`, or `sdft` |
| `data/` | Tooluse (ToolAlpaca, as used by SDFT) and LiveNews training sets |

## Setup

```bash
pip install torch==2.9.0 transformers==4.57.1
```

Every run fits on a single 80 GB GPU.

## Usage

```bash
python train.py --task tooluse --method gcd --model Qwen/Qwen2.5-3B-Instruct --output_dir runs/tooluse_gcd
```

The trained model and tokenizer are saved to `<output_dir>/final`, and the per-step loss to
`<output_dir>/train_log.jsonl`.
