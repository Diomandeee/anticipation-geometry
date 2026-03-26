#!/usr/bin/env python3
"""Filter raw Supabase turns into high-quality training records.

Applies aggressive quality gates:
  1. Drop short/empty turns (user < 20 chars, assistant < 50 chars)
  2. Drop repetitive content (dedup by content hash)
  3. Drop single-turn conversations (no trajectory dynamics)
  4. Drop low-salience conversations (all turns < 0.3 salience)
  5. Drop phase-flat conversations (same phase on every turn)
  6. Drop status-check noise ("continue", "what's the status", "ok")
  7. Keep only user→assistant pairs

Input: training/data/supabase_raw_turns.jsonl
Output: training/data/supabase_filtered.jsonl (ChatML messages format)

Usage:
    python3 filter_quality.py
    python3 filter_quality.py --min-conv-turns 4 --min-user-len 30
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

NOISE_PATTERNS = [
    r"^(ok|okay|yes|no|continue|thanks|thank you|got it|sure|done|next)\.?$",
    r"^what'?s the status",
    r"^check (the |on )?status",
    r"^continue\.?$",
    r"^\[Pasted text",
    r"^<task-notification",
    r"^<system-reminder",
]
NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

INSCRIPTION_VOCAB = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]


def is_noise(text: str) -> bool:
    """Check if text is a low-value status check or confirmation."""
    text = text.strip()
    if len(text) < 5:
        return True
    for pat in NOISE_RE:
        if pat.match(text):
            return True
    return False


def content_hash(text: str) -> str:
    """Hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())[:500]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def has_trajectory_dynamics(turns: list[dict]) -> bool:
    """Check if conversation has phase changes (not all same phase)."""
    phases = set(t.get("phase", "stabilization") for t in turns)
    return len(phases) >= 2


def conversation_salience(turns: list[dict]) -> float:
    """Mean salience across conversation turns."""
    saliences = [t.get("salience_score") or 0.5 for t in turns]
    return sum(saliences) / len(saliences) if saliences else 0.5


def compute_scalars(turns: list[dict]) -> dict:
    """Compute anticipation scalars from a conversation."""
    n = len(turns)
    saliences = [t.get("salience_score") or 0.5 for t in turns]
    phases = [t.get("phase", "stabilization") for t in turns]

    commitment = min(1.0, max(0.0, saliences[-1]))
    uncertainty = min(1.0, len(set(phases)) / 10)
    if len(saliences) >= 2:
        deltas = [saliences[i+1] - saliences[i] for i in range(len(saliences)-1)]
        tp = sum(deltas) / len(deltas)
    else:
        tp = 0.0
    recovery = max(0.0, 1.0 - commitment * 0.8)

    return {
        "commitment": round(commitment, 6),
        "uncertainty": round(uncertainty, 6),
        "transition_pressure": round(tp, 6),
        "recovery_margin": round(recovery, 6),
    }


def classify_inscription(phase: str, salience: float) -> str:
    if phase in INSCRIPTION_VOCAB:
        return phase
    if salience > 0.7:
        return "convergence"
    if salience < 0.3:
        return "stagnation"
    return "stabilization"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training/data/supabase_raw_turns.jsonl")
    parser.add_argument("--output", default="training/data/supabase_filtered.jsonl")
    parser.add_argument("--min-conv-turns", type=int, default=4,
                        help="Min turns per conversation to keep")
    parser.add_argument("--min-user-len", type=int, default=20,
                        help="Min user message length")
    parser.add_argument("--min-asst-len", type=int, default=50,
                        help="Min assistant message length")
    parser.add_argument("--min-salience", type=float, default=0.3,
                        help="Min mean conversation salience")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load raw turns
    print(f"Loading {input_path}...")
    convos = defaultdict(list)
    total_turns = 0
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            convos[t.get("conversation_id", "unknown")].append(t)
            total_turns += 1

    print(f"Raw: {total_turns} turns, {len(convos)} conversations")

    # Filter conversations
    stats = {
        "too_short": 0, "low_salience": 0, "no_dynamics": 0,
        "noise_turns": 0, "dedup_drops": 0, "short_content": 0,
        "kept": 0,
    }

    seen_hashes = set()
    records = []

    for cid, turns in convos.items():
        # Gate 1: conversation length
        if len(turns) < args.min_conv_turns:
            stats["too_short"] += 1
            continue

        # Gate 2: salience threshold
        if conversation_salience(turns) < args.min_salience:
            stats["low_salience"] += 1
            continue

        # Gate 3: trajectory dynamics (phase diversity)
        if not has_trajectory_dynamics(turns):
            stats["no_dynamics"] += 1
            continue

        # Compute scalars for this conversation
        scalars = compute_scalars(turns)

        # Extract user→assistant pairs
        n = len(turns)
        i = 0
        while i < n - 1:
            if turns[i].get("role") == "user" and i+1 < n and turns[i+1].get("role") == "assistant":
                ut = (turns[i].get("content_text", "") or "").strip()
                at = (turns[i+1].get("content_text", "") or "").strip()

                # Gate 4: noise filter
                if is_noise(ut):
                    stats["noise_turns"] += 1
                    i += 2
                    continue

                # Gate 5: length filter
                if len(ut) < args.min_user_len or len(at) < args.min_asst_len:
                    stats["short_content"] += 1
                    i += 2
                    continue

                # Gate 6: deduplication
                h = content_hash(ut + at)
                if h in seen_hashes:
                    stats["dedup_drops"] += 1
                    i += 2
                    continue
                seen_hashes.add(h)

                # Truncate very long messages
                ut = ut[:2000]
                at = at[:3000]

                position = round(i / max(n, 1), 4)
                inscription = classify_inscription(
                    turns[i+1].get("phase", "stabilization"),
                    turns[i+1].get("salience_score") or 0.5,
                )

                records.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": ut},
                        {"role": "assistant", "content": at},
                    ],
                    "scalars": scalars,
                    "inscription": inscription,
                    "position": position,
                })
                stats["kept"] += 1
                i += 2
            else:
                i += 1

    # Write output
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nFilter results:")
    print(f"  Conversations dropped (too short < {args.min_conv_turns} turns): {stats['too_short']}")
    print(f"  Conversations dropped (low salience < {args.min_salience}): {stats['low_salience']}")
    print(f"  Conversations dropped (no phase dynamics): {stats['no_dynamics']}")
    print(f"  Turns dropped (noise/status checks): {stats['noise_turns']}")
    print(f"  Turns dropped (short content): {stats['short_content']}")
    print(f"  Turns dropped (duplicate content): {stats['dedup_drops']}")
    print(f"  Records kept: {stats['kept']}")
    print(f"\nOutput: {output_path} ({len(records)} records)")

    # Inscription distribution
    inscriptions = defaultdict(int)
    for r in records:
        inscriptions[r["inscription"]] += 1
    print("\nInscription distribution:")
    for k, v in sorted(inscriptions.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(records):.1f}%)")


if __name__ == "__main__":
    main()
