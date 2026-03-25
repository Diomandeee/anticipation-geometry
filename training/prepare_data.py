#!/usr/bin/env python3
"""Prepare data for Vast.ai Anticipatory Transformer experiment.

Creates 4 JSONL files from existing KARL/anticipation-geometry data:
  1. real_conv_train.jsonl  — real conversation turns with scalars
  2. real_conv_eval.jsonl   — held-out eval set
  3. sft_106k.jsonl         — SFT records from KARL cascade
  4. synth_struct.jsonl     — synthetic structured examples

Each record has:
  {
    "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
    "scalars": {"commitment": float, "uncertainty": float, "transition_pressure": float, "recovery_margin": float},
    "inscription": str,   # behavioral motif label
    "position": float     # position in conversation [0,1]
  }
"""

import hashlib
import json
import random
import sys
from pathlib import Path

PROJ = Path(__file__).parent.parent
CURATED_TRAIN = PROJ / "training-data-curated" / "train.jsonl"
CURATED_VAL = PROJ / "training-data-curated" / "val.jsonl"
KARL_SFT = Path.home() / "projects" / "karl" / "data" / "merged" / "train_sft_ready.jsonl"
OUT_DIR = Path(__file__).parent / "data"

# Inscription vocabulary (10 sigils from KARL)
INSCRIPTIONS = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]


def classify_inscription(scalars: dict) -> str:
    """Heuristic inscription from scalar geometry."""
    c = scalars.get("commitment", 0.5)
    u = scalars.get("uncertainty", 0.5)
    tp = scalars.get("transition_pressure", 0.0)
    rm = scalars.get("recovery_margin", 0.5)

    if c > 0.8 and u < 0.3:
        return "convergence" if tp < 0.1 else "completion"
    if u > 0.7:
        return "oscillation" if tp > 0.3 else "exploration"
    if tp > 0.5:
        return "transition"
    if rm < 0.3:
        return "regression"
    if c > 0.6 and u < 0.4:
        return "stabilization"
    if tp < -0.2:
        return "stagnation"
    if c < 0.3:
        return "expansion" if u > 0.4 else "correction"
    return "stabilization"


def convert_curated(src: Path) -> list[dict]:
    """Convert curated training-data format to standard messages format."""
    records = []
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)

            scalars = d["scalars"] if isinstance(d["scalars"], dict) else json.loads(str(d["scalars"]).replace("'", '"'))
            position = float(d.get("position_in_conversation", 0.5))
            inscription = classify_inscription(scalars)

            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": d["input"]},
                {"role": "assistant", "content": d["target"]},
            ]

            records.append({
                "messages": messages,
                "scalars": {k: round(float(v), 6) for k, v in scalars.items()},
                "inscription": inscription,
                "position": round(position, 4),
            })
    return records


def convert_karl_sft(src: Path) -> list[dict]:
    """Convert KARL SFT records to standard format."""
    records = []
    if not src.exists():
        print(f"  KARL SFT not found at {src}, skipping")
        return records

    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            msgs = d.get("input", {}).get("messages", [])
            target = d.get("target", {}).get("response", "")
            quality = d.get("quality", {})

            if not msgs or not target:
                continue

            messages = []
            for m in msgs:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "assistant", "content": target})

            # Derive scalars from quality metadata
            spec = quality.get("specificity", 0.5) if isinstance(quality, dict) else 0.5
            scalars = {
                "commitment": round(min(spec * 1.2, 1.0), 6),
                "uncertainty": round(max(1.0 - spec, 0.0), 6),
                "transition_pressure": round(random.gauss(0.0, 0.15), 6),
                "recovery_margin": round(random.uniform(0.4, 1.0), 6),
            }
            inscription = classify_inscription(scalars)

            records.append({
                "messages": messages,
                "scalars": scalars,
                "inscription": inscription,
                "position": round(random.uniform(0.2, 0.9), 4),
            })
    return records


def generate_synthetic(n: int, seed: int = 42) -> list[dict]:
    """Generate synthetic structured examples with known scalar patterns."""
    random.seed(seed)
    records = []
    templates = [
        ("Explain {topic} step by step.", "Here's a step-by-step explanation of {topic}:\n\n1. "),
        ("Fix the bug in this {lang} code: {snippet}", "The issue is "),
        ("Refactor this function for readability.", "Here's the refactored version:\n\n"),
        ("What's the best approach for {task}?", "For {task}, I'd recommend:\n\n"),
        ("Debug: {error}", "This error occurs because "),
    ]
    topics = ["recursion", "async/await", "type safety", "error handling", "caching", "testing", "CI/CD"]
    langs = ["Python", "TypeScript", "Swift", "Rust", "Go"]
    tasks = ["rate limiting", "retry logic", "connection pooling", "logging", "monitoring"]

    for i in range(n):
        tmpl_user, tmpl_asst = random.choice(templates)
        user = tmpl_user.format(
            topic=random.choice(topics),
            lang=random.choice(langs),
            snippet="def f(x): return f(x)",
            task=random.choice(tasks),
            error="TypeError: cannot read property 'length' of undefined",
        )
        asst = tmpl_asst.format(
            topic=random.choice(topics),
            task=random.choice(tasks),
        ) + f"This is a structured synthetic example #{i}."

        # Synthetic scalars with known patterns
        pattern = random.choice(INSCRIPTIONS)
        if pattern == "convergence":
            scalars = {"commitment": 0.85 + random.gauss(0, 0.05), "uncertainty": 0.15 + random.gauss(0, 0.05), "transition_pressure": 0.02, "recovery_margin": 0.9}
        elif pattern == "oscillation":
            scalars = {"commitment": 0.4, "uncertainty": 0.75 + random.gauss(0, 0.05), "transition_pressure": 0.4, "recovery_margin": 0.5}
        else:
            scalars = {"commitment": random.uniform(0.2, 0.8), "uncertainty": random.uniform(0.2, 0.8), "transition_pressure": random.gauss(0, 0.2), "recovery_margin": random.uniform(0.3, 1.0)}

        scalars = {k: round(max(0, min(1, v)), 6) for k, v in scalars.items()}

        records.append({
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ],
            "scalars": scalars,
            "inscription": pattern,
            "position": round(random.uniform(0, 1), 4),
        })
    return records


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(records)} records")


def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Converting curated conversation data...")
    real_train = convert_curated(CURATED_TRAIN)
    real_eval = convert_curated(CURATED_VAL)

    print("2. Converting KARL SFT data...")
    sft_records = convert_karl_sft(KARL_SFT)

    print("3. Generating synthetic structured data...")
    # Target: 10% of total. If real=25K and sft=7K, total=32K, synth=3.5K
    n_synth = max(1000, int((len(real_train) + len(sft_records)) * 0.1 / 0.9))
    synth_records = generate_synthetic(n_synth)

    print(f"\n4. Writing datasets to {OUT_DIR}/")
    write_jsonl(real_train, OUT_DIR / "real_conv_train.jsonl")
    write_jsonl(real_eval, OUT_DIR / "real_conv_eval.jsonl")
    write_jsonl(sft_records, OUT_DIR / "sft_106k.jsonl")
    write_jsonl(synth_records, OUT_DIR / "synth_struct.jsonl")

    total = len(real_train) + len(sft_records) + len(synth_records)
    print(f"\nTotal training pool: {total}")
    print(f"  real_conv: {len(real_train)} ({100*len(real_train)/total:.1f}%)")
    print(f"  sft:       {len(sft_records)} ({100*len(sft_records)/total:.1f}%)")
    print(f"  synth:     {len(synth_records)} ({100*len(synth_records)/total:.1f}%)")
    print(f"  eval:      {len(real_eval)}")


if __name__ == "__main__":
    main()
