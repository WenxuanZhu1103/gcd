"""Fine-tune a model on Tooluse or LiveNews with SFT, SDFT, or GCD.

  sft   cross-entropy on the gold response
  gcd   forward KL to an EMA teacher that sees the gold response, evaluated along the gold response
  sdft  forward KL to an EMA teacher that sees the gold response, evaluated along a response the
        student samples itself (Shenfeld et al., 2026)

Usage: python train.py --task tooluse --method gcd --model Qwen/Qwen2.5-3B-Instruct --output_dir runs/tooluse_qwen25_gcd
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data import build_examples, data_path, load_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["tooluse", "livenews"], required=True)
    p.add_argument("--method", choices=["sft", "gcd", "sdft"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--ema_alpha", type=float, default=0.01, help="teacher <- (1 - a) teacher + a student")
    p.add_argument("--ce_tokens", type=int, default=1, help="GCD: first N response tokens use cross-entropy")
    p.add_argument("--skip_tokens", type=int, default=3, help="SDFT: first N sampled tokens are left out of the loss")
    p.add_argument("--max_new_tokens", type=int, default=1024, help="SDFT: sampling length limit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_chat_template", action="store_true", help="plain-text prompts; keep the model's own EOS")
    return p.parse_args()


def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    return model.cuda()


def setup_tokenizer(name, chat=True):
    tok = AutoTokenizer.from_pretrained(name)
    # Chat turns end with <|im_end|>; Qwen3-Base ships with <|endoftext|> as EOS, so realign it.
    # Plain-text prompts keep the pretraining EOS, the only end token a base model knows.
    if chat:
        tok.eos_token = "<|im_end|>"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    return tok


def response_logits(model, prompts, responses, pad_id):
    """Next-token logits at every response position, conditioned on prompt + response prefix.

    Prompts are left-padded and responses right-padded so all responses start at the same column.
    Returns logits (B, R, V) in fp32, the response token ids (B, R) and their mask (B, R).
    """
    P, R = max(map(len, prompts)), max(map(len, responses))
    ids = torch.full((len(prompts), P + R), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (p, r) in enumerate(zip(prompts, responses)):
        ids[i, P - len(p):P], mask[i, P - len(p):P] = torch.tensor(p), 1
        ids[i, P:P + len(r)], mask[i, P:P + len(r)] = torch.tensor(r), 1
    ids, mask = ids.cuda(), mask.cuda()
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=position_ids, logits_to_keep=R + 1).logits
    return logits[:, :-1].float(), ids[:, P:], mask[:, P:]


def forward_kl(teacher_logp, student_logp):
    """Per-position KL(teacher || student), summed over the vocabulary."""
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)


def sft_loss(student, batch, pad_id):
    logits, targets, mask = response_logits(student, batch["student_ids"], batch["response_ids"], pad_id)
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (ce * mask).sum() / mask.sum()  # mean over all response tokens in the batch


def gcd_loss(student, teacher, batch, pad_id, args):
    s_logits, targets, mask = response_logits(student, batch["student_ids"], batch["response_ids"], pad_id)
    with torch.no_grad():
        t_logits, _, _ = response_logits(teacher, batch["teacher_ids"], batch["response_ids"], pad_id)
    s_logp, t_logp = s_logits.log_softmax(-1), t_logits.log_softmax(-1)
    kl = forward_kl(t_logp, s_logp)
    ce = -s_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # The first token and the final EOS follow the response format rather than its content, and the
    # teacher's repeat prompt makes its distribution there unreliable, so they use cross-entropy.
    pos = torch.arange(mask.size(1), device=mask.device)
    is_eos = pos == (mask.sum(-1, keepdim=True) - 1)
    per_token = torch.where((pos < args.ce_tokens) | is_eos, ce, kl)
    return ((per_token * mask).sum(-1) / mask.sum(-1)).mean()


def sdft_loss(student, teacher, batch, pad_id, args):
    s_logits, _, mask = response_logits(student, batch["student_ids"], batch["sampled_ids"], pad_id)
    with torch.no_grad():
        t_logits, _, _ = response_logits(teacher, batch["teacher_ids"], batch["sampled_ids"], pad_id)
    kl = forward_kl(t_logits.log_softmax(-1), s_logits.log_softmax(-1))
    mask = mask * (torch.arange(mask.size(1), device=mask.device) >= args.skip_tokens)
    return ((kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()


@torch.no_grad()
def sample_responses(model, prompts, tok, max_new_tokens):
    """SDFT rollouts: sample one response per prompt from the current student (temperature 1).

    Sampling settings are passed explicitly so the model's generation_config (e.g. Qwen2.5-Instruct's
    repetition_penalty 1.05) does not change the student's distribution.
    """
    model.eval()  # turns off gradient checkpointing so generation can use the KV cache
    batch = tok.pad({"input_ids": prompts}, return_tensors="pt").to("cuda")
    out = model.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                         max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model.train()
    responses = []
    for seq in out[:, batch["input_ids"].size(1):].tolist():
        if tok.eos_token_id in seq:
            seq = seq[:seq.index(tok.eos_token_id) + 1]
        responses.append(seq)
    return responses


@torch.no_grad()
def ema_update(teacher, student, alpha):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.lerp_(s, alpha)


def make_optimizer(model, args):
    no_decay = lambda n: n.endswith("bias") or "norm" in n.lower()
    groups = [
        {"params": [p for n, p in model.named_parameters() if not no_decay(n)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if no_decay(n)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    chat = not args.no_chat_template
    tok = setup_tokenizer(args.model, chat)
    encode = lambda text: tok(text, add_special_tokens=False).input_ids
    examples = [
        {
            "student_ids": encode(ex["student_prompt"]),
            "teacher_ids": encode(ex["gcd_teacher_prompt" if args.method == "gcd" else "sdft_teacher_prompt"]),
            "response_ids": encode(ex["response"]) + [tok.eos_token_id],
        }
        for ex in build_examples(args.task, load_jsonl(data_path(args.task, "train")), tok, chat)
    ]

    student = load_model(args.model)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.train()
    teacher = None
    if args.method != "sft":
        teacher = load_model(args.model).eval().requires_grad_(False)

    # One optimizer step = grad_accum micro-batches; the data is reshuffled every epoch.
    micro = []
    for _ in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        micro += [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    steps = [micro[i:i + args.grad_accum] for i in range(0, len(micro) - args.grad_accum + 1, args.grad_accum)]

    optimizer = make_optimizer(student, args)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup_ratio * len(steps)), len(steps))
    log = open(os.path.join(args.output_dir, "train_log.jsonl"), "w")

    for step, step_batches in enumerate(steps, 1):
        if args.method == "sdft":
            idx = [i for b in step_batches for i in b]
            sampled = sample_responses(student, [examples[i]["student_ids"] for i in idx], tok, args.max_new_tokens)
            sampled = dict(zip(idx, sampled))
        total = 0.0
        for b in step_batches:
            batch = {k: [examples[i][k] for i in b] for k in examples[0]}
            if args.method == "sft":
                loss = sft_loss(student, batch, tok.pad_token_id)
            elif args.method == "gcd":
                loss = gcd_loss(student, teacher, batch, tok.pad_token_id, args)
            else:
                batch["sampled_ids"] = [sampled[i] for i in b]
                loss = sdft_loss(student, teacher, batch, tok.pad_token_id, args)
            (loss / len(step_batches)).backward()
            total += loss.item() / len(step_batches)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if teacher is not None:
            ema_update(teacher, student, args.ema_alpha)
        record = {"step": step, "loss": round(total, 5), "grad_norm": round(grad_norm, 3), "lr": scheduler.get_last_lr()[0]}
        log.write(json.dumps(record) + "\n")
        log.flush()
        if step % 10 == 0 or step == len(steps):
            print(record, flush=True)

    final_dir = os.path.join(args.output_dir, "final")
    student.config.use_cache = True
    student.generation_config.eos_token_id = tok.eos_token_id
    student.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)
    print("saved", final_dir)


if __name__ == "__main__":
    main()
