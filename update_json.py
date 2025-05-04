#!/usr/bin/env python3
import json
import logging
import os
import re
import time
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import requests

# ————————————————————————————————————————————————————————————————
# CONFIGURAZIONE & LOGGER
# ————————————————————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

WEIGHTS = {
    'topics':     0.6,
    'semantic':   0.2,
    'popularity': 0.15,
    'recency':    0.05,
}


# ————————————————————————————————————————————————————————————————
# 1) FETCH DEL JSON CON RETRY
# ————————————————————————————————————————————————————————————————
def fetch_json(url: str, retries: int = 3, backoff: int = 2) -> list[dict]:
    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return list(data.values()) if isinstance(data, dict) else data
        except Exception as e:
            logging.warning(f"[try {attempt}/{retries}] fetch error: {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    return []


# ————————————————————————————————————————————————————————————————
# 2) PREPROCESS E FEATURE ENGINEERING
# ————————————————————————————————————————————————————————————————
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # --- 2.1 colonne numeriche di base
    df['stars']       = df['stargazers_count'].fillna(0).astype(int)
    df['downloads']   = df['downloads'].fillna(0).astype(int)
    df['open_issues'] = df['open_issues'].fillna(0).astype(int)

    # --- 2.2 watchers (0 se non esiste)
    if 'watchers_count' not in df.columns:
        df['watchers_count'] = 0
    df['watchers_count'] = df['watchers_count'].fillna(0).astype(int)

    # --- 2.3 normalizzazioni [0,1]
    df[['stars_norm', 'downloads_norm', 'issues_norm', 'watchers_norm']] = (
        df[['stars', 'downloads', 'open_issues', 'watchers_count']]
        .div([
            df['stars'].max() or 1,
            df['downloads'].max() or 1,
            df['open_issues'].max() or 1,
            df['watchers_count'].max() or 1
        ], axis=1)
    )

    # --- 2.4 recency_norm
    df['last_updated'] = pd.to_datetime(df['last_updated'], utc=True)
    min_d, max_d = df['last_updated'].min(), df['last_updated'].max()
    span_days = max((max_d - min_d).days, 1)
    df['recency_norm'] = df['last_updated'].apply(lambda d: (d - min_d).days / span_days)

    # --- 2.5 development status
    now = pd.Timestamp.now(tz='UTC')
    df['age_days'] = (now - df['last_updated']).dt.days
    df['development_status'] = pd.cut(
        df['age_days'],
        bins=[-1, 30, 180, 365, float('inf')],
        labels=['active', 'maintained', 'inactive', 'abandoned']
    ).astype(str)

    # --- 2.6 top-download booleano
    q90 = df['downloads'].quantile(0.90)
    df['is_top_downloads'] = df['downloads'] >= q90

    # --- 2.7 popularity_score
    df['popularity_score'] = 0.5 * df['stars_norm'] + 0.5 * df['downloads_norm']

    # --- 2.8 metadati
    df['topic_count']        = df['topics'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['description_length'] = df['description'].fillna('').astype(str).str.len()

    # --- 2.9 top-100 flags per indicatore chiave
    numeric_inds = [
        'stars', 'downloads', 'watchers_count',
        'popularity_score', 'recency_norm', 'issues_norm'
    ]
    for ind in numeric_inds:
        flag_col = f'is_top100_{ind}'
        df[flag_col] = df[ind].rank(method='first', ascending=False) <= 100

    return df


# ————————————————————————————————————————————————————————————————
# 3) COSTRUZIONE SET DI TAG
# ————————————————————————————————————————————————————————————————
def make_sets(row: pd.Series) -> tuple[set, set]:
    # topics-only
    t = row.get('topics') or []
    topics_set = set(t) if isinstance(t, list) else {t} if t else set()
    # semantic = domain + tokenized description
    sem = set()
    if (d := row.get('domain')): sem.add(d)
    desc = row.get('description') or ''
    words = re.findall(r'\w+', str(desc).lower())
    sem |= {w for w in words if len(w) > 2}
    return topics_set, sem


# ————————————————————————————————————————————————————————————————
# 4) JACCARD SIMILARITY
# ————————————————————————————————————————————————————————————————
def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# ————————————————————————————————————————————————————————————————
# 5) RACCOMANDAZIONI IBRIDE
# ————————————————————————————————————————————————————————————————
def compute_recommendations(df: pd.DataFrame, top_k: int = 5) -> list[list[str]]:
    topics_sets, sem_sets = zip(*df.apply(make_sets, axis=1))
    pop_score = df['popularity_score'].tolist()
    rec_norm   = df['recency_norm'].tolist()
    names      = df['full_name'].tolist()

    recs = []
    for i in range(len(df)):
        scores = []
        for j in range(len(df)):
            if i == j:
                continue
            score = (
                WEIGHTS['topics']     * jaccard(topics_sets[i], topics_sets[j]) +
                WEIGHTS['semantic']   * jaccard(sem_sets[i],       sem_sets[j])   +
                WEIGHTS['popularity'] * pop_score[j]                                 +
                WEIGHTS['recency']    * rec_norm[j]
            )
            scores.append((j, score))
        top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        recs.append([names[j] for j, _ in top])
    return recs


# ————————————————————————————————————————————————————————————————
# 6) SALVATAGGIO ATOMICO E RIMOZIONE NaN
# ————————————————————————————————————————————————————————————————
def save_json(df: pd.DataFrame, out_path: str, atomically: bool = True):
    # sostituisci NaN/NA con None → verrà serializzato come null
    clean = df.where(pd.notnull(df), None).to_dict(orient='records')
    data = json.dumps(clean, ensure_ascii=False, separators=(',', ':'), default=str)
    if atomically:
        tmp = out_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(data)
        os.replace(tmp, out_path)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(data)
    logging.info(f"Wrote {len(clean)} records to {out_path}")


# ————————————————————————————————————————————————————————————————
# MAIN: orchestrazione
# ————————————————————————————————————————————————————————————————
def main():
    parser = ArgumentParser(description="Genera integrations_enhanced.json")
    parser.add_argument('--url',   default="https://data-v2.hacs.xyz/integration/data.json")
    parser.add_argument('--out',   default="docs/integrations_enhanced.json")
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()

    logging.info("Fetching data…")
    records = fetch_json(args.url)

    logging.info("Building DataFrame…")
    df = pd.json_normalize(records)
    # assicura stringhe non-null
    df['description'] = df['description'].fillna('').astype(str)
    df['domain']      = df['domain'].fillna('').astype(str)

    logging.info("Computing features…")
    df = preprocess(df)

    logging.info("Computing recommendations…")
    df['recommendations'] = compute_recommendations(df, top_k=args.top_k)

    logging.info("Saving JSON…")
    save_json(df, args.out)

    logging.info("Done.")


if __name__ == '__main__':
    main()
