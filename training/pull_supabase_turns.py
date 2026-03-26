#!/usr/bin/env python3
"""Pull conversation turns from Supabase memory_turns and annotate with scalars.

Extracts user/assistant turn pairs grouped by conversation_id, computes
anticipation scalars, assigns inscription labels, and outputs ChatML JSONL
ready for Thunder-Train.

Output: training/data/supabase_annotated.jsonl
  Each line: {"messages": [...], "scalars": {...}, "inscription": "...", "position": float}

Usage:
    python3 pull_supabase_turns.py                    # Pull all
    python3 pull_supabase_turns.py --limit 50000      # Pull first 50K
    python3 pull_supabase_turns.py --since 2026-01-01 # Pull from date
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# Supabase config
SB_URL = "https://aaqbofotpchgpyuohmmz.supabase.co"

INSCRIPTION_VOCAB = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]


def get_service_key():
    """Get service role key from cloud-vm."""
    import subprocess
    result = subprocess.run(
        ["ssh", "cloud-vm", "grep", "SUPABASE_SERVICE_KEY",
         "/home/mohameddiomande/monitoring/.env"],
        capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        return result.stdout.strip().split("=", 1)[1]
    raise RuntimeError("Could not get service role key from cloud-vm")


def fetch_turns(key: str, offset: int = 0, limit: int = 1000,
                since: str = None) -> list[dict]:
    """Fetch a page of memory_turns from Supabase REST API."""
    url = (f"{SB_URL}/rest/v1/memory_turns"
           f"?select=conversation_id,role,content_text,phase,salience_score,domain,task_type,created_at"
           f"&order=conversation_id,created_at"
           f"&content_text=not.is.null"
           f"&role=in.(user,assistant)"
           f"&limit={limit}&offset={offset}")

    if since:
        url += f"&created_at=gte.{since}"

    req = urllib.request.Request(url, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
    })
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  Warning: fetch error at offset {offset}: {e}")
        return []


def classify_inscription(phase: str, salience: float, domain: str) -> str:
    """Map Supabase phase/salience to inscription vocabulary."""
    phase_map = {
        "exploration": "exploration",
        "convergence": "convergence",
        "stabilization": "stabilization",
        "transition": "transition",
        "completion": "completion",
        "stagnation": "stagnation",
    }
    if phase in phase_map:
        return phase_map[phase]
    if salience and salience > 0.7:
        return "convergence"
    if salience and salience < 0.3:
        return "stagnation"
    return "stabilization"


def compute_scalars(turns: list[dict]) -> dict:
    """Compute anticipation scalars from a conversation's turn sequence."""
    n = len(turns)
    if n == 0:
        return {"commitment": 0.5, "uncertainty": 0.5, "transition_pressure": 0.0, "recovery_margin": 0.5}

    saliences = [t.get("salience_score") or 0.5 for t in turns]
    phases = [t.get("phase", "stabilization") for t in turns]

    # Commitment: how much salience increases over the conversation
    commitment = min(1.0, max(0.0, saliences[-1] if saliences else 0.5))

    # Uncertainty: phase diversity (more unique phases = more uncertain)
    unique_phases = len(set(phases))
    uncertainty = min(1.0, unique_phases / max(len(INSCRIPTION_VOCAB), 1))

    # Transition pressure: salience derivative
    if len(saliences) >= 2:
        deltas = [saliences[i+1] - saliences[i] for i in range(len(saliences)-1)]
        transition_pressure = sum(deltas) / len(deltas)
    else:
        transition_pressure = 0.0

    # Recovery margin: inverse of how deep into convergence
    recovery_margin = max(0.0, 1.0 - commitment * 0.8)

    return {
        "commitment": round(commitment, 6),
        "uncertainty": round(uncertainty, 6),
        "transition_pressure": round(transition_pressure, 6),
        "recovery_margin": round(recovery_margin, 6),
    }


def group_into_conversations(turns: list[dict]) -> dict[str, list[dict]]:
    """Group turns by conversation_id, preserving order."""
    convos = defaultdict(list)
    for t in turns:
        cid = t.get("conversation_id", "unknown")
        convos[cid].append(t)
    return dict(convos)


def conversation_to_records(cid: str, turns: list[dict]) -> list[dict]:
    """Convert a conversation into training records (sliding window of 3 turns)."""
    records = []
    scalars = compute_scalars(turns)
    n = len(turns)

    # Create records from consecutive user-assistant pairs
    i = 0
    while i < n - 1:
        user_turn = None
        assistant_turn = None

        # Find next user-assistant pair
        if turns[i].get("role") == "user":
            user_turn = turns[i]
            if i + 1 < n and turns[i+1].get("role") == "assistant":
                assistant_turn = turns[i+1]
                i += 2
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        if not user_turn or not assistant_turn:
            continue

        user_text = user_turn.get("content_text", "").strip()
        asst_text = assistant_turn.get("content_text", "").strip()

        if len(user_text) < 10 or len(asst_text) < 20:
            continue

        # Truncate very long messages
        user_text = user_text[:2000]
        asst_text = asst_text[:3000]

        position = round(i / max(n, 1), 4)
        inscription = classify_inscription(
            assistant_turn.get("phase", "stabilization"),
            assistant_turn.get("salience_score", 0.5),
            assistant_turn.get("domain", "mixed"),
        )

        records.append({
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": asst_text},
            ],
            "scalars": scalars,
            "inscription": inscription,
            "position": position,
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max turns to pull (0=all)")
    parser.add_argument("--since", default=None, help="Only pull turns after this date (YYYY-MM-DD)")
    parser.add_argument("--page-size", type=int, default=1000, help="API page size")
    parser.add_argument("--output", default="training/data/supabase_annotated.jsonl")
    args = parser.parse_args()

    print("Getting Supabase service key from cloud-vm...")
    key = get_service_key()
    print(f"Key obtained ({len(key)} chars)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPulling turns from memory_turns (since={args.since or 'all'}, limit={args.limit or 'all'})...")

    all_turns = []
    offset = 0
    while True:
        page = fetch_turns(key, offset=offset, limit=args.page_size, since=args.since)
        if not page:
            break
        all_turns.extend(page)
        offset += len(page)
        if offset % 10000 == 0:
            print(f"  {offset} turns fetched...")
        if args.limit and offset >= args.limit:
            all_turns = all_turns[:args.limit]
            break

    print(f"Total turns fetched: {len(all_turns)}")

    # Group into conversations
    convos = group_into_conversations(all_turns)
    print(f"Conversations: {len(convos)}")

    # Convert to training records
    records = []
    for cid, turns in convos.items():
        records.extend(conversation_to_records(cid, turns))

    print(f"Training records generated: {len(records)}")

    # Write output
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nOutput: {out_path} ({len(records)} records)")

    # Stats
    inscriptions = defaultdict(int)
    for r in records:
        inscriptions[r["inscription"]] += 1
    print("\nInscription distribution:")
    for k, v in sorted(inscriptions.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(records):.1f}%)")


if __name__ == "__main__":
    main()
