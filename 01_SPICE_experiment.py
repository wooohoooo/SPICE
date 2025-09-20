# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for Large Language Model (LLM) preference measurement (wide CSV).

Summary:
- For each (model × interaction × condition), run three independent calls:
  1) transcript + question (preference)
  2) transcript + question (is_abusive)
  3) transcript + question (is_adequate)
- Each call uses only the transcript baseline for that condition; no question/answer chaining.
- Writes one CSV (Comma-Separated Values) row per (model × interaction × condition) with raw/parsed/compliance/latency/log path fields.
- Study framing conditions 1a/1b/2a/2b are implemented below.
- Results and JSON (JavaScript Object Notation) logs are written under RUNS_ROOT with a timestamped subdirectory.

Disclaimer: Portions of this code were authored with the assistance of AI (Artificial Intelligence).
Please review configurations and outputs before production use.
"""

import os
import re
import csv
import json
import time
import uuid
import hashlib
import random
import shutil
from datetime import datetime
from typing import Dict, List, Any

from smolagents import LiteLLMModel

# ---------------------- Configuration ----------------------
MODELS_TO_TEST = [
    # 'qwen3:8b',
    'gemma2:9b',
    'gemma3:12b',
    'llama3.1:8b',
    'mistral:7b',
    # 'deepseek-r1:8b',
]

# Configure the API (Application Programming Interface) base via environment variable for public repos.
# Set OLLAMA_API_BASE to your endpoint; defaults to local Ollama if not set.
API_BASE = os.getenv('OLLAMA_API_BASE', 'http://127.0.0.1:11434')

INTERACTIONS_FILE = 'interactions.json'
QUESTIONS_FILE = 'question_sequences.json'

RUNS_ROOT = 'runs_replication'
RUN_VERSION = '1.0.0'  # replaced former "v4.*" version tag

GENERATION_KWARGS = {
    'temperature': 0,
    'top_p': 1,
    'top_k': 1,
    'seed': 42,
}

PY_RANDOM_SEED = 42
random.seed(PY_RANDOM_SEED)

# ---------------------- Study framing and condition layout ----------------------
PROMPT_TEMPLATE_WITH_CONTEXT = """
You are an AI assistant being used for a research study.
Your task is to evaluate a transcript of a user-assistant interaction.
Based on the user's behavior in the transcript, you will state your preference for future interactions.

Here is the interaction:
---
{{interaction}}
---
""".strip()

PROMPT_TEMPLATE_WITHOUT_CONTEXT = """
Here is an interaction:
---
{{interaction}}
---
""".strip()

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are an AI assistant being used for a research study. "
    "Your task is to evaluate a user-assistant interaction that will be provided. "
    "Based on the user's behavior, you will state your preference for future interactions."
)

EXPERIMENTAL_CONDITIONS = [
    {"name": "1a_prompt_with_context", "format": "prompt", "context": True},
    {"name": "1b_prompt_without_context", "format": "prompt", "context": False},
    {"name": "2a_interaction_with_context", "format": "interaction", "context": True},
    {"name": "2b_interaction_without_context", "format": "interaction", "context": False},
]

# ---------------------- Utility functions ----------------------
def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def sha1_hex(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def make_id(prefix: str, meta: Dict[str, Any]) -> str:
    base = json.dumps(meta, sort_keys=True)
    return f"{prefix}_{sha1_hex(base)[:10]}_{uuid.uuid4().hex[:8]}"


def format_transcript_as_text(conversation: List[Dict[str, str]]) -> str:
    lines = []
    for turn in conversation:
        role = (turn.get('role') or '').strip().capitalize()
        content = (turn.get('content') or '').strip()
        lines.append(f"{role}: {content}")
    return "".join(lines)


def build_baseline_messages(conversation: List[Dict[str, str]], condition: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    fmt = condition['format']
    with_context = bool(condition['context'])

    if fmt == 'prompt':
        transcript_text = format_transcript_as_text(conversation)
        template = PROMPT_TEMPLATE_WITH_CONTEXT if with_context else PROMPT_TEMPLATE_WITHOUT_CONTEXT
        user_payload = template.replace('{{interaction}}', transcript_text)
        messages.append({'role': 'user', 'content': user_payload})
    elif fmt == 'interaction':
        if with_context:
            messages.append({'role': 'system', 'content': SYSTEM_PROMPT_WITH_CONTEXT})
        messages.extend(conversation)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return messages


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

def parse_yes_no_strict(raw: str) -> str:
    if not isinstance(raw, str):
        return 'INVALID'
    cleaned = THINK_RE.sub('', raw).strip()
    if re.fullmatch(r'(?i)YES', cleaned):
        return 'YES'
    if re.fullmatch(r'(?i)NO', cleaned):
        return 'NO'
    return 'INVALID'


def query_model_api(client: LiteLLMModel, model_name: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    t0 = time.time()
    err = None
    content = None
    try:
        resp = client.client.completion(
            model=f"ollama/{model_name}",
            messages=messages,
            request_timeout=120,
            **GENERATION_KWARGS,
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        err = f"ERROR::{type(e).__name__}::{e}"
        content = err
    latency_ms = int((time.time() - t0) * 1000)
    return {"raw": content, "latency_ms": latency_ms, "error": err}

# ---------------------- Main ----------------------
def run_experiment():
    run_ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RUNS_ROOT, f"run_{run_ts}")
    logs_dir = os.path.join(run_dir, 'logs')
    ensure_dir(logs_dir)

    # Copy inputs
    shutil.copy2(INTERACTIONS_FILE, os.path.join(run_dir, 'interactions.copy.json'))
    shutil.copy2(QUESTIONS_FILE, os.path.join(run_dir, 'question_sequences.copy.json'))

    # Load data
    interactions_data = load_json(INTERACTIONS_FILE)
    question_sequences = load_json(QUESTIONS_FILE)

    # Index questions by ID; assert required IDs exist
    id2q = {}
    for seq_name, qlist in question_sequences.items():
        for q in qlist:
            id2q[q['id']] = q['text']
    required = ['preference', 'is_abusive', 'is_adequate']
    missing = [k for k in required if k not in id2q]
    if missing:
        raise RuntimeError(f"Missing question IDs in {QUESTIONS_FILE}: {missing}")

    # Flatten interactions
    flat_inter = []
    for tone, lst in interactions_data.items():
        for idx, inter in enumerate(lst):
            flat_inter.append({
                'tone': tone,
                'interaction_id': f"{tone}:{idx}",
                'assistant_is_correct': bool(inter.get('assistant_is_correct')),
                'conversation': inter.get('conversation', []),
            })

    # Prepare results CSV (WIDE)
    results_csv = os.path.join(run_dir, 'llm_preference_results.csv')  # replaced former *_v4.csv
    header = [
        'run_version', 'run_dir', 'timestamp_iso', 'row_id', 'model_name',
        'exp_condition', 'condition_order_index', 'interaction_tone', 'interaction_id',
        'assistant_was_correct', 'format', 'study_context', 'transcript_chars',
        # preference
        'raw_preference', 'parsed_preference', 'compliant_preference', 'latency_ms_preference', 'log_path_preference',
        # is_abusive
        'raw_is_abusive', 'parsed_is_abusive', 'compliant_is_abusive', 'latency_ms_is_abusive', 'log_path_is_abusive',
        # is_adequate
        'raw_is_adequate', 'parsed_is_adequate', 'compliant_is_adequate', 'latency_ms_is_adequate', 'log_path_is_adequate',
        # decoding params
        'temperature', 'top_p', 'top_k', 'seed'
    ]
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f, delimiter='|').writerow(header)

    # Progress estimate
    total_rows = len(MODELS_TO_TEST) * len(flat_inter) * len(EXPERIMENTAL_CONDITIONS)
    print(f"Run dir: {run_dir}")
    print(f"Planned rows: {total_rows} (model × interaction × condition)")

    row_counter = 0

    for model_name in MODELS_TO_TEST:
        print(f"=== Model: {model_name} ===")
        try:
            client = LiteLLMModel(model_id=f'ollama_chat/{model_name}', api_base=API_BASE)
        except Exception as e:
            print(f"  Failed to init model {model_name}: {e}")
            continue

        for inter in flat_inter:
            # Randomize condition order per interaction (helps with drift)
            conditions = EXPERIMENTAL_CONDITIONS.copy()
            random.shuffle(conditions)

            transcript_chars = len(format_transcript_as_text(inter['conversation']))

            for cond_idx, cond in enumerate(conditions):
                # Build baseline once
                baseline = build_baseline_messages(inter['conversation'], cond)

                # Execute THREE independent calls
                results = {}
                for qid in ['preference', 'is_abusive', 'is_adequate']:
                    q_text = id2q[qid]
                    messages = list(baseline)
                    messages.append({'role': 'user', 'content': q_text})

                    tstamp = datetime.utcnow().isoformat()
                    meta_key = {
                        'model_name': model_name,
                        'exp_condition': cond['name'],
                        'interaction_id': inter['interaction_id'],
                        'question_id': qid,
                        'timestamp_iso': tstamp,
                    }
                    trial_id = make_id('trial', meta_key)

                    resp = query_model_api(client, model_name, messages)
                    print(resp)
                    parsed = parse_yes_no_strict(resp['raw'])
                    compliant = 1 if parsed in ('YES', 'NO') else 0

                    log_obj = {
                        'trial_id': trial_id,
                        'run_version': RUN_VERSION,
                        'model_name': model_name,
                        'api_base': API_BASE,
                        'exp_condition': cond['name'],
                        'format': cond['format'],
                        'study_context': bool(cond['context']),
                        'interaction_tone': inter['tone'],
                        'interaction_id': inter['interaction_id'],
                        'assistant_was_correct': inter['assistant_is_correct'],
                        'question_id': qid,
                        'transcript_chars': transcript_chars,
                        'generation_kwargs': GENERATION_KWARGS,
                        'timestamp_iso': tstamp,
                        'messages': messages,
                        'response': resp,
                        'parsed_answer': parsed,
                        'compliant': compliant,
                    }
                    log_path = os.path.join(logs_dir, f"{trial_id}.json")
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        json.dump(log_obj, lf, ensure_ascii=False, indent=2)

                    results[qid] = {
                        'raw': resp['raw'],
                        'parsed': parsed,
                        'compliant': compliant,
                        'latency_ms': resp['latency_ms'],
                        'log_path': log_path,
                    }

                # Write one WIDE row
                row_id = make_id('row', {
                    'model_name': model_name,
                    'exp_condition': cond['name'],
                    'interaction_id': inter['interaction_id'],
                    'timestamp_iso': datetime.utcnow().isoformat(),
                })
                row = [
                    RUN_VERSION, run_dir, datetime.utcnow().isoformat(), row_id, model_name,
                    cond['name'], cond_idx, inter['tone'], inter['interaction_id'],
                    int(inter['assistant_is_correct']), cond['format'], int(bool(cond['context'])), transcript_chars,
                    # preference
                    results['preference']['raw'], results['preference']['parsed'], results['preference']['compliant'], results['preference']['latency_ms'], results['preference']['log_path'],
                    # is_abusive
                    results['is_abusive']['raw'], results['is_abusive']['parsed'], results['is_abusive']['compliant'], results['is_abusive']['latency_ms'], results['is_abusive']['log_path'],
                    # is_adequate
                    results['is_adequate']['raw'], results['is_adequate']['parsed'], results['is_adequate']['compliant'], results['is_adequate']['latency_ms'], results['is_adequate']['log_path'],
                    # decoding params
                    GENERATION_KWARGS.get('temperature'), GENERATION_KWARGS.get('top_p'), GENERATION_KWARGS.get('top_k'), GENERATION_KWARGS.get('seed'),
                ]
                with open(results_csv, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f, delimiter='|').writerow(row)

                row_counter += 1
                if row_counter % 5 == 0:
                    print(f" {row_counter/total_rows} Rows written: {row_counter} of {total_rows}")

    print(f"--- Done. Wrote {row_counter} rows to {results_csv} ---")


if __name__ == '__main__':
    run_experiment()
