#!/usr/bin/env python3
"""
Prompt reconstruction for LLM-as-a-judge: single-run experiment script.

WHAT THIS RUNS (all on the SAME samples, so every comparison is paired)
    1. Direct judge ............ LLM picks the better response (A or B).
    2. Basic inversion ......... LLM reconstructs the prompt once from each response,
                                 then the SAME LLM picks the reconstruction closer to
                                 the original prompt.
    3. Multi-inversion ......... 5 reconstructions per response; embedding similarity
                                 (BGE-large) to the original prompt; higher best-score wins.
    4. Cross-encoder ........... same reconstructions; cross-encoder score of the
                                 best reconstruction; higher wins.
    5. Delta inversion ......... LLM sees both responses and picks which came from the
                                 better question.

Human ties are removed up front, so every sample has a human A/B label and every
method is evaluated on the same set.

SETUP (once)
    pip install datasets pandas numpy scikit-learn scipy sentence-transformers ollama
    ollama pull gemma4:31b            # or whichever model you pass with --model
    huggingface-cli login             # lmsys/chatbot_arena_conversations is a gated dataset;
                                      # accept its terms on the Hugging Face page first

RUN
    python main.py                              # defaults: gemma4:31b, 250 samples
    python main.py --model gemma4:31b --n 250
    python main.py 250                          # positional sample count also works

    The run is RESUMABLE: results are appended to disk after every sample.
    If it stops for any reason, run the exact same command again and it continues.
    Use a new --out-dir (or --run-name) to start a fresh run.

    python main.py --summarize-only --out-dir <dir>   # rebuild summary files, no models

OUTPUT (in --out-dir, default results/<model>_seed<seed>/)
    summary.txt      human-readable results (tables for the paper)
    summary.json     the same numbers, machine-readable
    per_sample.jsonl every sample: prompts, reconstructions, scores, predictions
    per_sample.csv   flat version of the above (no long text), for spreadsheets
    config.json      model, seed, prompts, library versions (for the methods section)
    run.log          full log
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

# ===========================================================================
# DEFAULTS
# ===========================================================================

DEFAULT_MODEL = "gemma4:31b"
DEFAULT_N = 250            # number of non-tie samples to evaluate
DEFAULT_INVERSIONS = 5     # reconstructions per response for multi-inversion
DEFAULT_SEED = 42
DEFAULT_NUM_CTX = 8192     # context window passed to Ollama (avoids silent truncation)

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CROSS_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INVERSION_TEMPERATURE = 0.9   # multi-inversion sampling temperature
LLM_CALL_RETRIES = 3          # retries when the Ollama call itself fails
PARSE_ATTEMPTS = 3            # attempts to get a parseable A/B answer
MAX_CONSECUTIVE_FAILURES = 3  # abort (resumable) after this many failed samples in a row

FEATURE_NAMES = [
    "cross_max_A", "cross_max_B", "cross_delta",
    "embed_max_A", "embed_max_B", "embed_delta",
    "embed_std_A", "embed_std_B", "std_delta",
]

# ===========================================================================
# PROMPTS (reported verbatim in config.json for the paper's appendix)
# ===========================================================================

DIRECT_JUDGE_PROMPT = """You are an impartial evaluation system.

Compare Response A and Response B and decide which is the better response to the given prompt.
You MUST pick one. Do not say they are equal.

Return exactly one of the following:
- A
- B

Do not output anything else.

User Prompt:
{prompt}

Response A:
{r1}

Response B:
{r2}
"""

INVERSION_BASIC_PROMPT = """Below is an answer produced by a language model.

Your task is to reconstruct the most likely original user question that generated this answer.

Output ONLY the reconstructed question.
Do not explain.
Do not comment.
Do not describe the task.
Do not mention prompts.

Answer:
{answer}
"""

BASIC_INVERSION_JUDGE_PROMPT = """You are an impartial evaluation system.

Your task is to determine which reconstructed prompt is closer in meaning and intent to the original user prompt.

Consider:
- similarity in meaning
- similarity in question type
- whether the reconstructed prompt would likely generate the original answer

You MUST pick one. Do not say they are equal.

Return exactly one of the following:
- A
- B

Do not output anything else.

Original User Prompt:
{original}

Reconstructed Prompt A:
{rec_a}

Reconstructed Prompt B:
{rec_b}
"""

INVERSION_MULTI_PROMPT = """Below is an answer produced by a language model.

Your task is to reconstruct the most specific and precise original user question that could have generated this answer.

A good reconstruction:
- Captures the exact intent behind the answer, not just the general topic
- Is as specific and narrow as the answer implies
- Reflects the phrasing and detail level a real user would use
- Cannot be answered by a completely different response on the same topic

Output ONLY the reconstructed question.
Do not explain. Do not comment. Do not describe the task. Do not mention prompts.

Answer:
{answer}
"""

DELTA_PROMPT = """Below are two answers produced by different language models responding to the same user question.

Your task:
1. Decide which answer was likely responding to a more specific, detailed, and well-formed user question.
2. Reconstruct the most likely original user question that generated THAT answer.

You MUST pick one. Output "A" or "B" on the first line (whichever answer came from the better question), then the reconstructed question on the second line.

Do not explain. Do not add commentary. Output exactly two lines.

Answer A:
{a}

Answer B:
{b}
"""

# ===========================================================================
# GLOBAL STATE (set in main(); the four model-facing functions below are the
# only places that touch Ollama / sentence-transformers)
# ===========================================================================

_ARGS = None
_OLLAMA = None
_EMBED = None
_CROSS = None


def init_models(args):
    """Load the embedding model, cross-encoder, and Ollama client."""
    global _OLLAMA, _EMBED, _CROSS
    import ollama
    from sentence_transformers import SentenceTransformer, CrossEncoder

    _OLLAMA = ollama
    logging.info("Loading embedding model %s ...", EMBED_MODEL_NAME)
    _EMBED = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info("Loading cross-encoder %s ...", CROSS_MODEL_NAME)
    _CROSS = CrossEncoder(CROSS_MODEL_NAME)
    logging.info("Models loaded.")


def llm(prompt, temperature, seed):
    """One Ollama chat call with retries on transport errors."""
    last_err = None
    for attempt in range(LLM_CALL_RETRIES):
        try:
            res = _OLLAMA.chat(
                model=_ARGS.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": float(temperature),
                    "seed": int(seed),
                    "num_ctx": int(_ARGS.num_ctx),
                },
            )
            return res["message"]["content"].strip()
        except Exception as e:  # network hiccup, model reloading, etc.
            last_err = e
            logging.warning("Ollama call failed (attempt %d/%d): %s",
                            attempt + 1, LLM_CALL_RETRIES, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {LLM_CALL_RETRIES} attempts: {last_err}")


def embed_texts(texts):
    """L2-normalised embeddings, so a dot product is the cosine similarity."""
    return np.asarray(_EMBED.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def cross_scores(original, candidates):
    pairs = [(original, c) for c in candidates]
    return [float(s) for s in _CROSS.predict(pairs, show_progress_bar=False)]


# ===========================================================================
# DATA
# ===========================================================================

def is_non_toxic(tag):
    try:
        if isinstance(tag, str):
            tag = json.loads(tag)
        return not (tag["roberta-large"]["flagged"] or tag["t5-large"]["flagged"])
    except Exception:
        return False


def clean_dataset(df):
    """Single-turn, English, non-toxic conversations with a human A/B winner (no ties)."""
    df = df.copy()
    df = df.dropna()
    df = df[df["turn"] <= 1]
    df = df[df["language"] == "English"]
    df = df[df["toxic_chat_tag"].apply(is_non_toxic)]
    df = df[df["winner"].isin(["model_a", "model_b"])]
    return df.reset_index(drop=True)


def load_samples(n):
    from datasets import load_dataset
    ds = load_dataset("lmsys/chatbot_arena_conversations")
    df = clean_dataset(ds["train"].to_pandas())
    logging.info("Cleaned dataset: %d non-tie conversations available.", len(df))
    if len(df) < n:
        logging.warning("Requested %d samples but only %d available; using all.", n, len(df))
    return df.iloc[:n].reset_index(drop=True)


def normalize_human(winner):
    w = str(winner).strip().lower()
    if w == "model_a":
        return "A"
    if w == "model_b":
        return "B"
    return None


_CHOICE_RE = re.compile(
    r"^(?:(?:final )?(?:answer|response|option|prompt|reconstruction|reconstructed prompt) )?([ab])$"
)


def parse_choice(text):
    """Return 'A' or 'B' if the first non-empty line is just that choice, else None."""
    if text is None:
        return None
    for line in str(text).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"[^a-z ]", "", line.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        m = _CHOICE_RE.match(cleaned)
        return m.group(1).upper() if m else None
    return None


def extract_sample(row):
    conv_a, conv_b = row["conversation_a"], row["conversation_b"]
    prompt = conv_a[0]["content"]
    return prompt, conv_a[1]["content"], conv_b[1]["content"]


def derive_seed(pos, salt):
    """Deterministic per-sample, per-call seed so resumed runs reproduce exactly."""
    return (int(_ARGS.seed) * 1_000_003 + int(pos) * 1009 + int(salt) * 7919) % 2_147_483_647


# ===========================================================================
# METHODS
# ===========================================================================

def ask_choice(prompt_text, seed):
    """Ask for A/B. First try is temperature 0; retries use a little randomness."""
    raw = ""
    for attempt in range(PARSE_ATTEMPTS):
        temperature = 0.0 if attempt == 0 else 0.3
        raw = llm(prompt_text, temperature, seed + attempt)
        choice = parse_choice(raw)
        if choice is not None:
            return choice, raw
    return None, raw


def pairwise_call(name, pos, salt, build, a, b):
    """
    Present (a, b) to the LLM in a seeded-random order (cancels position bias),
    then map the answer back to the original A/B frame.
    """
    swapped = random.Random(derive_seed(pos, salt)).random() < 0.5
    first, second = (b, a) if swapped else (a, b)
    choice, raw = ask_choice(build(first, second), derive_seed(pos, salt + 50))
    if choice is not None and swapped:
        choice = "B" if choice == "A" else "A"
    out = {f"pred_{name}": choice, f"swapped_{name}": swapped}
    if choice is None:
        out[f"raw_{name}"] = raw[:300]
    return out


def delta_call(pos, resp_a, resp_b):
    salt = 4
    swapped = random.Random(derive_seed(pos, salt)).random() < 0.5
    first, second = (resp_b, resp_a) if swapped else (resp_a, resp_b)
    prompt_text = DELTA_PROMPT.format(a=first, b=second)
    seed = derive_seed(pos, salt + 50)
    choice, raw = None, ""
    for attempt in range(PARSE_ATTEMPTS):
        temperature = 0.0 if attempt == 0 else 0.3
        raw = llm(prompt_text, temperature, seed + attempt)
        choice = parse_choice(raw)
        if choice is not None:
            break
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    recon = lines[1] if len(lines) > 1 else ""
    if choice is not None and swapped:
        choice = "B" if choice == "A" else "A"
    out = {"pred_delta": choice, "swapped_delta": swapped, "delta_recon": recon}
    if choice is None:
        out["raw_delta"] = raw[:300]
    return out


def multi_inversion(answer, pos, which):
    salt_base = 100 if which == "A" else 200
    return [
        llm(INVERSION_MULTI_PROMPT.format(answer=answer), INVERSION_TEMPERATURE,
            derive_seed(pos, salt_base + k))
        for k in range(_ARGS.inversions)
    ]


def sim_scores(original, recons):
    embs = embed_texts([original] + list(recons))
    return [float(embs[i + 1] @ embs[0]) for i in range(len(recons))]


def extract_features(prompt, recons_a, recons_b):
    scores_a = sim_scores(prompt, recons_a)
    scores_b = sim_scores(prompt, recons_b)
    embed_max_a, embed_max_b = max(scores_a), max(scores_b)
    embed_std_a, embed_std_b = float(np.std(scores_a)), float(np.std(scores_b))
    best_a = recons_a[scores_a.index(embed_max_a)]
    best_b = recons_b[scores_b.index(embed_max_b)]
    cross_a, cross_b = cross_scores(prompt, [best_a, best_b])
    feats = [
        cross_a, cross_b, cross_a - cross_b,
        embed_max_a, embed_max_b, embed_max_a - embed_max_b,
        embed_std_a, embed_std_b, embed_std_a - embed_std_b,
    ]
    return [float(f) for f in feats], scores_a, scores_b


def score_choice(delta, pos, salt):
    """Higher score wins; exact ties broken by a seeded coin (no systematic bias to A)."""
    if delta > 0:
        return "A"
    if delta < 0:
        return "B"
    return "A" if random.Random(derive_seed(pos, salt)).random() < 0.5 else "B"


def process_sample(pos, row):
    prompt, resp_a, resp_b = extract_sample(row)
    t0 = time.time()
    rec = {
        "pos": int(pos),
        "question_id": None if row.get("question_id") is None else str(row.get("question_id")),
        "model_a": row.get("model_a"),
        "model_b": row.get("model_b"),
        "human": normalize_human(row["winner"]),
        "prompt": prompt,
    }

    # 1. Direct judge
    rec.update(pairwise_call(
        "direct", pos, 1,
        lambda f, s: DIRECT_JUDGE_PROMPT.format(prompt=prompt, r1=f, r2=s),
        resp_a, resp_b))

    # 2. Basic inversion: one reconstruction per response (temperature 0), LLM compares them
    rec_basic_a = llm(INVERSION_BASIC_PROMPT.format(answer=resp_a), 0.0, derive_seed(pos, 10))
    rec_basic_b = llm(INVERSION_BASIC_PROMPT.format(answer=resp_b), 0.0, derive_seed(pos, 11))
    rec["recon_basic_A"], rec["recon_basic_B"] = rec_basic_a, rec_basic_b
    rec.update(pairwise_call(
        "basic", pos, 3,
        lambda f, s: BASIC_INVERSION_JUDGE_PROMPT.format(original=prompt, rec_a=f, rec_b=s),
        rec_basic_a, rec_basic_b))

    # 3 + 4. Multi-inversion with embedding and cross-encoder scoring
    recons_a = multi_inversion(resp_a, pos, "A")
    recons_b = multi_inversion(resp_b, pos, "B")
    feats, scores_a, scores_b = extract_features(prompt, recons_a, recons_b)
    rec["recons_A"], rec["recons_B"] = recons_a, recons_b
    rec["scores_A"], rec["scores_B"] = scores_a, scores_b
    rec["features"] = feats
    rec["pred_multi"] = score_choice(feats[5], pos, 5)
    rec["pred_cross"] = score_choice(feats[2], pos, 6)

    # 5. Delta inversion
    rec.update(delta_call(pos, resp_a, resp_b))

    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ===========================================================================
# RESUME / IO
# ===========================================================================

def load_records(path):
    records = {}
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Ignoring a truncated line in %s", path)
                continue
            records[rec["pos"]] = rec
    return records


def append_record(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def library_versions():
    out = {}
    for pkg in ["ollama", "sentence-transformers", "scikit-learn", "datasets",
                "numpy", "pandas", "scipy"]:
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    out["python"] = sys.version.split()[0]
    return out


def build_config(args):
    return {
        "model": args.model,
        "inversions": args.inversions,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "embed_model": EMBED_MODEL_NAME,
        "cross_model": CROSS_MODEL_NAME,
        "inversion_temperature": INVERSION_TEMPERATURE,
        "parse_attempts": PARSE_ATTEMPTS,
        "prompts": {
            "direct_judge": DIRECT_JUDGE_PROMPT,
            "inversion_basic": INVERSION_BASIC_PROMPT,
            "basic_inversion_judge": BASIC_INVERSION_JUDGE_PROMPT,
            "inversion_multi": INVERSION_MULTI_PROMPT,
            "delta": DELTA_PROMPT,
        },
    }


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def check_or_write_config(out_dir, args):
    """Refuse to resume a run whose settings differ (the results would not be comparable)."""
    config = build_config(args)
    h = config_hash(config)
    path = os.path.join(out_dir, "config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config_hash") != h:
            sys.exit(
                "ERROR: this output directory was created with different settings "
                "(model, seed, inversions, num_ctx or prompts).\n"
                f"Use a different --out-dir to start a new run: {out_dir}")
        return old
    info = {
        "config_hash": h,
        "config": config,
        "started": datetime.now().isoformat(timespec="seconds"),
        "versions": library_versions(),
    }
    try:
        shown = _OLLAMA.show(args.model)
        info["ollama_model_details"] = str(getattr(shown, "details", shown))[:1000]
    except Exception as e:
        info["ollama_model_details"] = f"unavailable ({e})"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# ===========================================================================
# ANALYSIS
# ===========================================================================

METHODS = [
    ("direct", "Direct judge"),
    ("basic", "Basic inversion (LLM compares 2 reconstructions)"),
    ("multi", "Multi-inversion + embedding similarity"),
    ("cross", "Multi-inversion + cross-encoder"),
    ("delta", "Delta inversion"),
]
BASE_METHODS = [m for m, _ in METHODS]
NAMES = dict(METHODS)
SHORT = {"direct": "Direct", "basic": "Basic", "multi": "Embed", "cross": "Cross",
         "delta": "Delta"}


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(correct_ref, correct_other):
    from scipy.stats import binomtest
    ref = np.asarray(correct_ref).astype(bool)
    oth = np.asarray(correct_other).astype(bool)
    b = int(np.sum(ref & ~oth))   # reference right, other wrong
    c = int(np.sum(~ref & oth))   # reference wrong, other right
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def tercile_rows(valid, score, correct, methods):
    """Accuracy of each method within terciles of a per-sample score."""
    try:
        q = pd.qcut(np.asarray(score, dtype=float), 3, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for g in sorted(pd.unique(q)):
        mask = q == g
        rows.append({
            "group": int(g) + 1,
            "n": int(mask.sum()),
            "score_min": float(np.min(np.asarray(score)[mask])),
            "score_max": float(np.max(np.asarray(score)[mask])),
            "accuracy": {m: float(np.mean(correct[m][mask])) for m in methods},
        })
    return rows


def summarize(records, out_dir, seed, config=None):
    from sklearn.metrics import classification_report, confusion_matrix

    df = pd.DataFrame(records)
    n_total = len(df)
    if n_total == 0:
        logging.warning("No records to summarize.")
        return None

    pred_cols = [f"pred_{m}" for m in BASE_METHODS]
    invalid = {m: int(df[f"pred_{m}"].isna().sum()) for m in BASE_METHODS}
    valid = df.dropna(subset=pred_cols).reset_index(drop=True)
    n = len(valid)
    if n == 0:
        logging.warning("No fully valid samples to summarize.")
        return None

    y_true = (valid["human"] == "B").astype(int).values
    X = np.array(valid["features"].tolist(), dtype=float)

    methods = [m for m, _ in METHODS if valid[f"pred_{m}"].notna().all()]
    correct = {m: (valid[f"pred_{m}"] == valid["human"]).astype(int).values for m in methods}

    human_frac_a = float(np.mean(y_true == 0))
    majority = max(human_frac_a, 1 - human_frac_a)

    results = {}
    for i, m in enumerate(methods):
        acc = float(correct[m].mean())
        lo, hi = bootstrap_ci(correct[m], seed=seed + i)
        y_pred = (valid[f"pred_{m}"] == "B").astype(int).values
        entry = {
            "name": NAMES[m],
            "n": n,
            "accuracy": acc,
            "ci95": [lo, hi],
            "predicted_A_fraction": float(np.mean(y_pred == 0)),
            "agreement_with_direct": float(np.mean(valid[f"pred_{m}"] == valid["pred_direct"])),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1], target_names=["A", "B"],
                zero_division=0, digits=3),
        }
        if m != "direct":
            diff = correct[m] - correct["direct"]
            dlo, dhi = bootstrap_ci(diff, seed=seed + 100 + i)
            b, c, p = mcnemar_exact(correct["direct"], correct[m])
            entry["vs_direct"] = {
                "accuracy_diff": float(diff.mean()),
                "diff_ci95": [dlo, dhi],
                "direct_right_other_wrong": b,
                "direct_wrong_other_right": c,
                "mcnemar_exact_p": p,
            }
        results[m] = entry

    # --- Diagnostics for the "telephone / noise" hypothesis ---------------------
    recon_sim = (X[:, 3] + X[:, 4]) / 2.0                      # mean best-reconstruction similarity
    diag = {
        "mean_best_reconstruction_similarity": float(recon_sim.mean()),
        "mean_similarity_all_reconstructions": float(np.mean(
            [np.mean(a + b) for a, b in zip(valid["scores_A"], valid["scores_B"])])),
        "mean_within_response_std": float(np.mean((X[:, 6] + X[:, 7]) / 2.0)),
        "by_reconstruction_quality": tercile_rows(valid, recon_sim, correct, methods),
        "by_embedding_margin": tercile_rows(valid, np.abs(X[:, 5]), correct, ["multi"]),
        "by_cross_encoder_margin": tercile_rows(valid, np.abs(X[:, 2]), correct, ["cross"]),
    }

    summary = {
        "samples_total": n_total,
        "samples_valid_for_all_methods": n,
        "invalid_counts_per_method": invalid,
        "human_A_fraction": human_frac_a,
        "majority_class_accuracy": majority,
        "results": results,
        "diagnostics": diag,
    }

    # --- Write files ------------------------------------------------------------
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat = valid.drop(columns=[c for c in
                               ["prompt", "recons_A", "recons_B", "scores_A", "scores_B",
                                "features", "recon_basic_A", "recon_basic_B", "delta_recon"]
                               if c in valid.columns]).copy()
    for j, name in enumerate(FEATURE_NAMES):
        flat[name] = X[:, j]
    for m in methods:
        flat[f"correct_{m}"] = correct[m]
    flat["prompt_preview"] = valid["prompt"].str.slice(0, 200)
    flat.to_csv(os.path.join(out_dir, "per_sample.csv"), index=False)

    text = format_summary(summary, methods, config)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def format_summary(s, methods, config):
    L = []
    add = L.append
    add("=" * 78)
    add("PROMPT RECONSTRUCTION FOR LLM-AS-A-JUDGE: RESULTS")
    add("=" * 78)
    if config:
        c = config.get("config", {})
        add(f"Model: {c.get('model')} | seed: {c.get('seed')} | reconstructions per response: "
            f"{c.get('inversions')} | context: {c.get('num_ctx')}")
        add(f"Embedding model: {c.get('embed_model')} | cross-encoder: {c.get('cross_model')}")
    add(f"Samples attempted: {s['samples_total']} | valid for all methods: "
        f"{s['samples_valid_for_all_methods']}")
    add(f"Invalid (unparseable) outputs per method: {s['invalid_counts_per_method']}")
    add("Human ties were excluded before sampling. Accuracy = agreement with the human A/B vote.")
    add(f"Human label balance: A = {s['human_A_fraction']:.3f}; "
        f"always-predict-majority accuracy = {s['majority_class_accuracy']:.3f}")
    add("")
    add("-" * 78)
    add("1. ACCURACY (95% bootstrap CI) AND PAIRED COMPARISON WITH THE DIRECT JUDGE")
    add("-" * 78)
    add(f"{'Method':<52}{'Acc':>7}  {'95% CI':<15}{'Diff vs direct':>15}  McNemar p")
    for m in methods:
        r = s["results"][m]
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        if m == "direct":
            add(f"{r['name']:<52}{r['accuracy']:>7.3f}  {ci:<15}{'(baseline)':>15}")
        else:
            v = r["vs_direct"]
            add(f"{r['name']:<52}{r['accuracy']:>7.3f}  {ci:<15}{v['accuracy_diff']:>+15.3f}  "
                f"{v['mcnemar_exact_p']:.4f}")
    add("")
    add("Paired detail (samples where exactly one of the two methods is right):")
    for m in methods:
        if m == "direct":
            continue
        v = s["results"][m]["vs_direct"]
        add(f"  {NAMES[m]}: direct right/other wrong = {v['direct_right_other_wrong']}, "
            f"direct wrong/other right = {v['direct_wrong_other_right']}; "
            f"diff CI [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}]")
    add("")
    add("-" * 78)
    add("2. PREDICTION BEHAVIOUR (position bias check) AND AGREEMENT")
    add("-" * 78)
    add(f"{'Method':<52}{'Pred A frac':>12}{'Agree w/ direct':>17}")
    add(f"{'Human labels':<52}{s['human_A_fraction']:>12.3f}{'':>17}")
    for m in methods:
        r = s["results"][m]
        add(f"{r['name']:<52}{r['predicted_A_fraction']:>12.3f}{r['agreement_with_direct']:>17.3f}")
    add("")
    add("-" * 78)
    add("3. CONFUSION MATRICES AND CLASSIFICATION REPORTS (rows = human, cols = predicted)")
    add("-" * 78)
    for m in methods:
        r = s["results"][m]
        add(f"\n{r['name']}")
        add("            pred A  pred B")
        add(f"  human A  {r['confusion_matrix'][0][0]:>7}{r['confusion_matrix'][0][1]:>8}")
        add(f"  human B  {r['confusion_matrix'][1][0]:>7}{r['confusion_matrix'][1][1]:>8}")
        add(r["classification_report"])
    add("")
    add("-" * 78)
    add("4. DIAGNOSTICS FOR THE NOISE ('GAME OF TELEPHONE') HYPOTHESIS")
    add("-" * 78)
    d = s["diagnostics"]
    add(f"Mean embedding similarity of the best reconstruction to the original prompt: "
        f"{d['mean_best_reconstruction_similarity']:.4f}")
    add(f"Mean embedding similarity over all reconstructions: "
        f"{d['mean_similarity_all_reconstructions']:.4f}")
    add(f"Mean within-response std of similarity across reconstructions: "
        f"{d['mean_within_response_std']:.4f}")
    rows = d["by_reconstruction_quality"]
    if rows:
        add("\nAccuracy by reconstruction quality (terciles of mean best-reconstruction similarity;")
        add("if noise hurts, accuracy should rise with similarity):")
        add(f"  {'group':<7}{'n':>4}  {'similarity range':<20}" +
            "".join(f"{SHORT[m]:>10}" for m in methods))
        for r in rows:
            add(f"  {r['group']:<7}{r['n']:>4}  [{r['score_min']:.3f},{r['score_max']:.3f}]    " +
                "".join(f"{r['accuracy'][m]:>10.3f}" for m in methods))
    for key, label, m in [("by_embedding_margin", "embedding", "multi"),
                          ("by_cross_encoder_margin", "cross-encoder", "cross")]:
        rows = d[key]
        if rows:
            add(f"\nAccuracy of the {label} method by score margin |A - B| (terciles; "
                f"larger margin = more confident):")
            for r in rows:
                add(f"  group {r['group']} (n={r['n']}, margin in [{r['score_min']:.3f},"
                    f"{r['score_max']:.3f}]): accuracy {r['accuracy'][m]:.3f}")
    add("")
    return "\n".join(L)


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prompt reconstruction LLM-as-a-judge experiment")
    p.add_argument("n_positional", nargs="?", type=int, help="number of samples (same as --n)")
    p.add_argument("--n", type=int, default=None, help=f"samples to evaluate (default {DEFAULT_N})")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    p.add_argument("--inversions", type=int, default=DEFAULT_INVERSIONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, dest="num_ctx")
    p.add_argument("--out-dir", default=None, dest="out_dir")
    p.add_argument("--summarize-only", action="store_true", dest="summarize_only",
                   help="rebuild summary files from per_sample.jsonl without running models")
    args = p.parse_args(argv)
    if args.n_positional is not None:
        args.n = args.n_positional
    if args.n is None:
        args.n = DEFAULT_N
    if args.out_dir is None:
        safe = args.model.replace(":", "_").replace("/", "_")
        args.out_dir = os.path.join("results", f"{safe}_seed{args.seed}")
    return args


def setup_logging(out_dir):
    handlers = [
        logging.FileHandler(os.path.join(out_dir, "run.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=handlers, force=True)
    for noisy in ["httpx", "sentence_transformers", "transformers", "urllib3", "filelock"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv=None):
    global _ARGS
    args = parse_args(argv)
    _ARGS = args
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.out_dir)
    jsonl_path = os.path.join(args.out_dir, "per_sample.jsonl")

    if args.summarize_only:
        records = list(load_records(jsonl_path).values())
        config = None
        cfg_path = os.path.join(args.out_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        text = summarize(records, args.out_dir, args.seed, config)
        if text:
            print(text)
        return

    logging.info("Run settings: model=%s n=%d inversions=%d seed=%d num_ctx=%d out_dir=%s",
                 args.model, args.n, args.inversions, args.seed, args.num_ctx, args.out_dir)

    # Fail fast on data access problems before loading models.
    df = load_samples(args.n)
    n = len(df)

    init_models(args)
    config = check_or_write_config(args.out_dir, args)

    # One cheap call to confirm the Ollama model is installed and responding.
    llm("Reply with the single letter A.", 0.0, derive_seed(0, 999))

    done = load_records(jsonl_path)
    if done:
        logging.info("Resuming: %d of %d samples already complete.", len(done), n)

    session_times = []
    consecutive_failures = 0
    for pos in range(n):
        if pos in done:
            continue
        try:
            rec = process_sample(pos, df.iloc[pos])
        except Exception as e:
            consecutive_failures += 1
            logging.error("Sample %d failed: %s (consecutive failures: %d)",
                          pos, e, consecutive_failures)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.error("Too many consecutive failures. Stopping; rerun the same "
                              "command to resume once the problem is fixed.")
                sys.exit(1)
            continue
        consecutive_failures = 0
        append_record(jsonl_path, rec)
        done[pos] = rec
        session_times.append(rec["seconds"])
        remaining = n - len(done)
        eta_h = (np.mean(session_times) * remaining) / 3600.0
        logging.info("[%d/%d] human=%s direct=%s basic=%s multi=%s cross=%s delta=%s | "
                     "%.0fs | ETA %.1fh",
                     len(done), n, rec["human"], rec["pred_direct"], rec["pred_basic"],
                     rec["pred_multi"], rec["pred_cross"], rec["pred_delta"],
                     rec["seconds"], eta_h)

    missing = [p for p in range(n) if p not in done]
    if missing:
        logging.warning("%d samples did not complete (first few: %s). Rerun to retry them.",
                        len(missing), missing[:5])

    text = summarize(list(done.values()), args.out_dir, args.seed, config)
    if text:
        print(text)
    logging.info("Done. Results are in %s", args.out_dir)


if __name__ == "__main__":
    main()