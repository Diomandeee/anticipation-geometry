#!/usr/bin/env python3
"""Prepare data variants for Thunder-Train 5-run experiment.

Creates ChatML JSONL files with/without inscription prefixes for
controlled comparison. Each file has the same conversations but
different conditioning formats.

Output (on Mac5 at ~/projects/atx-data/):
  sft_standard.jsonl          — control: plain ChatML, no inscription
  sft_inscription.jsonl       — treatment: inscription + scalars prepended
  sft_inscription_nogate.jsonl — ablation: inscription only, no gate metadata
  sft_noinscrip_gate.jsonl    — ablation: gate metadata only, no inscription
  real_conv_eval.jsonl         — eval set (standard format)
"""

import json
import random
import hashlib
from pathlib import Path

VASTAI_DATA = Path(__file__).parent / "data"
OUT_DIR = Path("/tmp/atx-thunder-data")

INSCRIPTION_VOCAB = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]


def load_all_sources():
    """Load and merge all data sources."""
    records = []

    # Real conversation data
    for path in [VASTAI_DATA / "real_conv_train.jsonl", VASTAI_DATA / "sft_106k.jsonl",
                 VASTAI_DATA / "synth_struct.jsonl"]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)
    return records


def to_chatml(record, mode="standard"):
    """Convert a record to ChatML format with optional inscription prefix.

    Modes:
      standard          — plain messages, no metadata
      inscription       — [MOTIF:xxx] prefix + scalar line before user message
      inscription_nogate — inscription prefix only, no gate/commitment info
      noinscrip_gate    — gate/commitment metadata only, no inscription
    """
    messages = record["messages"]
    scalars = record.get("scalars", {})
    inscription = record.get("inscription", "stabilization")
    commitment = scalars.get("commitment", 0.5)

    # Build messages list (Thunder-Train expects {"messages": [...]})
    sys_content = messages[0]["content"] if messages else "You are a helpful assistant."

    if mode == "inscription":
        sys_content = (f"[MOTIF:{inscription}] "
                       f"commitment={commitment:.2f} uncertainty={scalars.get('uncertainty', 0.5):.2f} "
                       f"transition={scalars.get('transition_pressure', 0.0):.2f}\n"
                       f"{sys_content}")
    elif mode == "inscription_nogate":
        sys_content = f"[MOTIF:{inscription}]\n{sys_content}"
    elif mode == "noinscrip_gate":
        sys_content = (f"commitment={commitment:.2f} "
                       f"uncertainty={scalars.get('uncertainty', 0.5):.2f}\n"
                       f"{sys_content}")

    out_messages = [{"role": "system", "content": sys_content}]
    for msg in messages[1:]:
        out_messages.append({"role": msg["role"], "content": msg["content"]})

    return {"messages": out_messages}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_all_sources()
    print(f"Total records: {len(records)}")

    modes = {
        "sft_standard.jsonl": "standard",
        "sft_inscription.jsonl": "inscription",
        "sft_inscription_nogate.jsonl": "inscription_nogate",
        "sft_noinscrip_gate.jsonl": "noinscrip_gate",
    }

    for filename, mode in modes.items():
        out_path = OUT_DIR / filename
        with open(out_path, "w") as f:
            for r in records:
                chatml = to_chatml(r, mode=mode)
                f.write(json.dumps(chatml) + "\n")
        print(f"  {filename}: {len(records)} records")

    # Eval set (standard format)
    eval_path = VASTAI_DATA / "real_conv_eval.jsonl"
    if eval_path.exists():
        eval_records = []
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    eval_records.append(json.loads(line))
        out_eval = OUT_DIR / "real_conv_eval.jsonl"
        with open(out_eval, "w") as f:
            for r in eval_records:
                f.write(json.dumps(to_chatml(r, mode="standard")) + "\n")
        print(f"  real_conv_eval.jsonl: {len(eval_records)} records")

    print(f"\nAll data at {OUT_DIR}/")
    print("Copy to Mac5: rsync -a /tmp/atx-thunder-data/ mac5:~/projects/atx-data/")


if __name__ == "__main__":
    main()
