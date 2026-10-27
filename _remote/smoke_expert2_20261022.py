"""Raw-audio experts, v2 (2026-10-22). Derived from smoke_g1expert_20261021.py (cbf7ba8a..., unchanged); the e2 recipe
is identical. Changes:
  e3  : WavLM Large (models/wavlm-large, microsoft/wavlm-large rev c1423ed9, native WavLMModel) with the e2 recipe;
        25-layer weighted sum, head width 1024. Card: runs/C15KD与闸门V结果_20261022/冲SOTA方案与E3卡_2026-10-22.md §4.
  MELD: raw wav from /root/autodl-tmp/meld_raw/<split>/<clip_id>.wav (16 kHz mono 16-bit, extracted by the user from
        the official MELD videos). Utterances without a wav (MELD train dia125_utt3, dev dia110_utt7) are left out of
        training and get a uniform log-probability (the expert abstains) in dev / test, flagged "no_audio".
  e1 / e1w are dropped (gate 1 closed them); use the old script for them.
Selection: final epoch on IEMOCAP (as gate 1); best development epoch on MELD (as gate 1 for MELD), unless --select.

Original docstring of smoke_g1expert_20261021.py:
Gate 1 audio experts (2026-10-21; card runs/闸门1_音频专家_卡_2026-10-20.md §2.1-2.3). One script, three experts:
  e2  : WavLM Base+ fine-tuned (CNN feature encoder frozen, transformer trained), 13-layer softmax-weighted sum
        (weights start at 0), attention pooling over valid frames, dropout 0.1, linear -> classes.
        Encoder lr 3e-5, head + layer weights 1e-3, AdamW wd 0.01, clip 1.0, 10% linear warm-up then cosine to 0,
        8 epochs, batch 8, bf16 autocast, no SpecAugment (mask_time_prob 0), no gradient checkpointing.
  e1w : the same with the encoder frozen (eval + no_grad); only head + layer weights train (lr 1e-3).
  e1  : offline emotion2vec frame features of the target utterance (64 x 768, all-zero frames masked), attention
        pooling, dropout 0.1, linear; lr 1e-3, wd 0.01, warm-up + cosine, 30 epochs, batch 32.
Raw audio (e2, e1w): IEMOCAP wav, 16 kHz mono 16-bit (stdlib wave; no package changes), first 15.0 s of every
utterance. Sentence -> wav mapping from runs/iemocap_align_20261018.json. Labels and keys from the dataset module, so
the expert outputs align with the text-only dumps (key = dialogue_id|utt_id, class order = spec.label_list).
Selection: final epoch on IEMOCAP; best development epoch on MELD (e1 only; the card allows no raw-audio expert there).
Outputs in --out_dir: config.json, metrics_dev.json, metrics_test.json, preds_dev.jsonl, preds_test.jsonl (key, gold,
pred, log-probabilities), dev_curve.json, the checkpoint (e2: full bf16 weights; e1w / e1: head only) with sha256.
--smoke: first 200 training utterances, 1 epoch, first 100 dev / test utterances; the test subset is scored twice and
must give identical predictions; reports peak memory, seconds per step and truncation counts. With --smoke, --expert
may list several experts (comma separated); each writes to <out_dir>/<expert>/.
Named _remote/smoke_* only so that the resident queue counts it as a GPU slot."""
import argparse, hashlib, json, math, os, random, sys, time, wave
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(8)  # 2026-10-21 17:45: three parallel E1 runs oversubscribed the CPU (171 threads each, ~6x slower than alone); CPU side is data prep only, the expert runs on GPU
sys.path.insert(0, os.getcwd())
from dataset_spec import get_spec  # noqa: E402

WAVLM = {"e2": "models/wavlm-base-plus", "e3": "models/wavlm-large"}
MELD_WAV = "/root/autodl-tmp/meld_raw"
WAV_ROOT = "/root/iemocap_raw/x/iemocap_raw"
ALIGN = "runs/iemocap_align_20261018.json"
SR, MAX_SEC = 16000, 15.0
CFG = {"e2": dict(epochs=8, batch=8, enc_lr=3e-5, head_lr=1e-3), "e3": dict(epochs=8, batch=8, enc_lr=3e-5, head_lr=1e-3)}
WD, CLIP, WARMUP = 0.01, 1.0, 0.10


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, (path, w.getparams())
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype="<i2")
    trunc = len(x) > int(SR * MAX_SEC)
    return x[: int(SR * MAX_SEC)].copy(), trunc


def wf1(y, p, C):
    out = 0.0
    for c in range(C):
        n = int((y == c).sum())
        if n == 0:
            continue
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum())
        out += n / len(y) * (2 * tp / (2 * tp + fp + n - tp) if tp else 0.0)
    return 100 * out


class AttnPool(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, H, mask):
        """H (B, T, d) float32, mask (B, T) bool. Padded / masked frames get weight exactly 0."""
        sc = self.score(H).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(sc, dim=1)
        w = torch.nan_to_num(w, nan=0.0) * mask          # an utterance with no valid frame pools to 0
        return (w.unsqueeze(-1) * H).sum(1), w


class Head(nn.Module):
    def __init__(self, n_layers, n_cls, d=768):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(n_layers)) if n_layers > 1 else None
        self.pool = AttnPool(d)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(d, n_cls)

    def forward(self, hs, mask):
        if self.lw is not None:
            a = torch.softmax(self.lw, 0)
            H = sum(a[i] * hs[i].float() for i in range(len(hs)))
        else:
            H = hs.float()
        z, _ = self.pool(H, mask)
        return self.fc(self.drop(z))


def load_split_items(spec, D, split, expert, align, feats):
    samples = D.load_split(split)
    items = []
    for s in samples:
        it = {"key": f"{s['dialogue_id']}|{s['utt_id']}", "dialogue_id": s["dialogue_id"], "utt_id": s["utt_id"], "y": s["label_id"]}
        if align is not None:
            it["wav"] = os.path.join(WAV_ROOT, align[f"{s['dialogue_id']}|{s['utt_id']}"]["wav"])
        else:
            it["wav"] = os.path.join(MELD_WAV, split, f"{s['clip_id']}.wav")
        it["no_audio"] = not os.path.exists(it["wav"])
        items.append(it)
    return items


class Expert(nn.Module):
    def __init__(self, expert, n_cls):
        super().__init__()
        self.expert = expert
        from transformers import WavLMConfig, WavLMModel
        cfg = WavLMConfig.from_pretrained(WAVLM[expert])
        cfg.mask_time_prob, cfg.apply_spec_augment = 0.0, False
        self.enc = WavLMModel.from_pretrained(WAVLM[expert], config=cfg)
        self.enc.freeze_feature_encoder()
        self.head = Head(cfg.num_hidden_layers + 1, n_cls, d=cfg.hidden_size)

    def forward(self, batch):
        x, am = batch["x"], batch["am"]
        out = self.enc(x, attention_mask=am, output_hidden_states=True)
        hs = out.hidden_states
        mask = self.enc._get_feature_vector_attention_mask(hs[0].shape[1], am).bool()
        return self.head(hs, mask)


def make_batch(items, expert, cache, device):
    wavs = [cache[it["key"]] for it in items]
    L = max(len(w) for w in wavs)
    x = np.zeros((len(wavs), L), dtype=np.float32); am = np.zeros((len(wavs), L), dtype=np.int64)
    for i, w in enumerate(wavs):
        x[i, : len(w)] = w.astype(np.float32) / 32768.0; am[i, : len(w)] = 1
    return {"x": torch.from_numpy(x).to(device), "am": torch.from_numpy(am).to(device)}


@torch.no_grad()
def predict(model, items, expert, cache, device, bs):
    model.eval()
    C = model.head.fc.out_features
    L = np.full((len(items), C), -np.log(C), dtype=np.float32)   # no_audio rows: uniform (the expert abstains)
    idx = [i for i, it in enumerate(items) if not it["no_audio"]]
    for i in range(0, len(idx), bs):
        b = [items[j] for j in idx[i:i + bs]]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(make_batch(b, expert, cache, device))
        L[idx[i:i + bs]] = torch.log_softmax(logits.float(), -1).cpu().numpy()
    return L


def run(expert, a, spec, D, out_dir, align):
    os.makedirs(out_dir, exist_ok=True)
    seed_all(a.seed)
    C = spec.num_classes
    cfg = dict(CFG[expert]); epochs = 1 if a.smoke else cfg["epochs"]
    splits = {sp: load_split_items(spec, D, sp, expert, align, None) for sp in ("train", "dev", "test")}
    if a.smoke:
        splits = {"train": splits["train"][:200], "dev": splits["dev"][:100], "test": splits["test"][:100]}
    cache, trunc = {}, {}
    for sp, its in splits.items():
        n = 0
        for it in its:
            if not it["no_audio"]:
                cache[it["key"]], t = read_wav(it["wav"]); n += t
        trunc[sp] = {"truncated_at_15s": n, "n": len(its), "no_audio": [it["key"] for it in its if it["no_audio"]]}
    splits["train"] = [it for it in splits["train"] if not it["no_audio"]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Expert(expert, C).to(dev)
    groups = [{"params": list(model.head.parameters()), "lr": cfg["head_lr"]}]
    groups.append({"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": cfg["enc_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=WD)
    tr = splits["train"]; bs = cfg["batch"]
    steps_per_epoch = len(tr) // bs; total = steps_per_epoch * epochs; warm = int(WARMUP * total)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, warm, total)
    g = torch.Generator().manual_seed(a.seed)
    curve, best = [], None
    t0 = time.time(); nstep = 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g).tolist()
        for i in range(steps_per_epoch):
            b = [tr[j] for j in perm[i * bs:(i + 1) * bs]]
            yb = torch.tensor([it["y"] for it in b], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(make_batch(b, expert, cache, dev))
            loss = nn.functional.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for gr in groups for p in gr["params"]], CLIP)
            opt.step(); sched.step(); nstep += 1
        Ld = predict(model, splits["dev"], expert, cache, dev, bs)
        ydv = np.array([it["y"] for it in splits["dev"]])
        dwf = wf1(ydv, Ld.argmax(1), C)
        curve.append({"epoch": ep, "dev_wf1": round(dwf, 3), "last_loss": round(float(loss), 4)})
        print(f"[{expert}] epoch {ep} dev wF1 {dwf:.2f} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if a.select == "best" and (best is None or dwf > best[0] + 1e-9):
            best = (dwf, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
    sec_per_step = (time.time() - t0) / max(nstep, 1)
    if a.select == "best":
        model.load_state_dict(best[2], strict=False)
    chosen = epochs - 1 if a.select == "last" else best[1]
    res = {}
    for sp in ("dev", "test"):
        L = predict(model, splits[sp], expert, cache, dev, bs)
        y = np.array([it["y"] for it in splits[sp]]); p = L.argmax(1)
        res[sp] = {"weighted_f1": round(wf1(y, p, C), 3), "accuracy": round(100 * float((y == p).mean()), 3), "n": len(y)}
        with open(os.path.join(out_dir, f"preds_{sp}.jsonl"), "w") as f:
            for it, lp, pp in zip(splits[sp], L, p):
                f.write(json.dumps({"key": it["key"], "dialogue_id": it["dialogue_id"], "utt_id": it["utt_id"],
                                    "gold": it["y"], "pred": int(pp), "logp": [round(float(v), 6) for v in lp],
                                    "no_audio": it["no_audio"]}) + "\n")
        json.dump(res[sp], open(os.path.join(out_dir, f"metrics_{sp}.json"), "w"), indent=1)
    if a.smoke:   # determinism: score the same subset again with the same weights
        L2 = predict(model, splits["test"], expert, cache, dev, bs)
        L1 = np.array([json.loads(l)["logp"] for l in open(os.path.join(out_dir, "preds_test.jsonl"))])
        res["determinism"] = {"preds_identical": bool((L1.argmax(1) == L2.argmax(1)).all()),
                              "max_abs_logp_diff": float(np.abs(L1 - L2).max())}
    sd = {k: (v.bfloat16() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    ck = os.path.join(out_dir, "expert_bf16.pt")
    torch.save(sd, ck)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    conf = {"expert": expert, "dataset": a.dataset, "seed": a.seed, "smoke": a.smoke, "select": a.select,
            "chosen_epoch": chosen, "epochs": epochs, **cfg, "weight_decay": WD, "clip": CLIP, "warmup": WARMUP,
            "max_sec": MAX_SEC, "n": {k: len(v) for k, v in splits.items()}, "input_report": trunc,
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev.type == "cuda" else None,
            "sec_per_step": round(sec_per_step, 4), "checkpoint": os.path.basename(ck), "sha256_checkpoint": sha(ck),
            "sha256_script": sha(os.path.abspath(__file__)), "label_list": list(spec.label_list),
            "wavlm": WAVLM[expert], "wavlm_sha256_pytorch_model": sha(os.path.join(WAVLM[expert], "pytorch_model.bin")),
            "results": res}
    json.dump(curve, open(os.path.join(out_dir, "dev_curve.json"), "w"), indent=1)
    json.dump(conf, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    print(f"[{expert}] done: dev {res['dev']['weighted_f1']:.2f} test {res['test']['weighted_f1']:.2f} "
          f"chosen epoch {chosen} | {conf['peak_mem_mb']} MB, {sec_per_step:.3f} s/step | {trunc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--dataset", default="iemocap", choices=["iemocap", "meld"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--select", default=None, choices=["last", "best"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_dir", default=os.environ.get("GRACE_OUTPUT_DIR"))
    a = ap.parse_args()
    a.select = a.select or ("last" if a.dataset == "iemocap" else "best")
    experts = a.expert.split(",")
    assert len(experts) == 1 or a.smoke, "several experts only with --smoke"
    spec = get_spec(a.dataset); D = spec.get_data_module()
    align = json.load(open(ALIGN))["utts"] if a.dataset == "iemocap" else None
    for e in experts:
        assert e in CFG, e
        run(e, a, spec, D, a.out_dir if len(experts) == 1 else os.path.join(a.out_dir, e), align)


if __name__ == "__main__":
    main()
