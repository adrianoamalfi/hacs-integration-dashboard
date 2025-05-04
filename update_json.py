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
    # 2.1 colonne numeriche
    df['stars']       = df['stargazers_count'].fillna(0).astype(int)
    df['downloads']   = df['downloads'].fillna(0).astype(int)
    df['open_issues'] = df['open_issues'].fillna(0).astype(int)
    if 'watchers_count' not in df.columns:
        df['watchers_count'] = 0
    df['watchers_count'] = df['watchers_count'].fillna(0).astype(int)

    # 2.2 normalizzazioni
    df[['stars_norm','downloads_norm','issues_norm','watchers_norm']] = (
        df[['stars','downloads','open_issues','watchers_count']]
        .div([
            df['stars'].max() or 1,
            df['downloads'].max() or 1,
            df['open_issues'].max() or 1,
            df['watchers_count'].max() or 1
        ], axis=1)
    )

    # 2.3 recency_norm
    df['last_updated'] = pd.to_datetime(df['last_updated'], utc=True)
    min_d, max_d = df['last_updated'].min(), df['last_updated'].max()
    span_days = max((max_d - min_d).days, 1)
    df['recency_norm'] = df['last_updated'].apply(lambda d: (d - min_d).days / span_days)

    # 2.4 development_status
    now = pd.Timestamp.now(tz='UTC')
    df['age_days'] = (now - df['last_updated']).dt.days
    df['development_status'] = pd.cut(
        df['age_days'],
        bins=[-1,30,180,365,float('inf')],
        labels=['active','maintained','inactive','abandoned']
    ).astype(str)

    # 2.5 flags e punteggi
    q90 = df['downloads'].quantile(0.90)
    df['is_top_downloads'] = df['downloads'] >= q90
    df['popularity_score'] = 0.5*df['stars_norm'] + 0.5*df['downloads_norm']
    df['topic_count']       = df['topics'].apply(lambda x: len(x) if isinstance(x,list) else 0)
    df['description_length']= df['description'].fillna('').astype(str).str.len()

    # 2.6 top-100 flags
    numeric_inds = ['stars','downloads','watchers_count','popularity_score','recency_norm','issues_norm']
    for ind in numeric_inds:
        df[f'is_top100_{ind}'] = df[ind].rank(method='first', ascending=False) <= 100

    return df


# ————————————————————————————————————————————————————————————————
# 3) TAG SET & JACCARD
# ————————————————————————————————————————————————————————————————
def make_sets(row: pd.Series):
    t = row.get('topics') or []
    topics_set = set(t) if isinstance(t,list) else {t} if t else set()
    sem = set()
    if row.get('domain'): sem.add(row['domain'])
    desc = row.get('description') or ''
    words = re.findall(r'\w+', str(desc).lower())
    sem |= {w for w in words if len(w)>2}
    return topics_set, sem

def jaccard(a:set,b:set)->float:
    return len(a&b)/len(a|b) if (a or b) else 0.0


# ————————————————————————————————————————————————————————————————
# 4) RACCOMANDAZIONI IBRIDE
# ————————————————————————————————————————————————————————————————
def compute_recommendations(df: pd.DataFrame, top_k:int):
    topics_sets, sem_sets = zip(*df.apply(make_sets,axis=1))
    pop, rec = df['popularity_score'].tolist(), df['recency_norm'].tolist()
    names = df['full_name'].tolist()
    recs=[]
    for i in range(len(df)):
        scores=[]
        for j in range(len(df)):
            if i==j: continue
            s = (WEIGHTS['topics']* jaccard(topics_sets[i],topics_sets[j])
               + WEIGHTS['semantic']* jaccard(sem_sets[i],    sem_sets[j])
               + WEIGHTS['popularity']*pop[j]
               + WEIGHTS['recency']   *rec[j])
            scores.append((j,s))
        top = sorted(scores, key=lambda x:x[1], reverse=True)[:top_k]
        recs.append([names[j] for j,_ in top])
    return recs


# ————————————————————————————————————————————————————————————————
# 5) SALVATAGGIO ATOMICO E RIMOZIONE NaN
# ————————————————————————————————————————————————————————————————
def save_json(df: pd.DataFrame, out_path: str, fields=None, limit=None):
    # 1) elimina i NaN
    df_clean = df.where(pd.notnull(df), None)
    # 2) seleziona solo i campi richiesti
    if fields:
        df_clean = df_clean.loc[:, fields]
    # 3) applica il limite sul numero di record
    if limit:
        df_clean = df_clean.sort_values('stars', ascending=False).head(limit)
    records = df_clean.to_dict(orient='records')
    data = json.dumps(records, ensure_ascii=False, separators=(',', ':'), default=str)
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(data)
    os.replace(tmp, out_path)
    logging.info(f"Wrote {len(records)} records to {out_path}")


# ————————————————————————————————————————————————————————————————
# MAIN
# ————————————————————————————————————————————————————————————————
def main():
    p = ArgumentParser()
    p.add_argument('--url',            default="https://data-v2.hacs.xyz/integration/data.json")
    p.add_argument('--out-full',       default="docs/integrations_full.json")
    p.add_argument('--out-summary',    default="docs/integrations_summary.json")
    p.add_argument('--top-k',          type=int, default=5)
    p.add_argument('--summary-limit',  type=int, default=100)
    args = p.parse_args()

    logging.info("Fetching data…")
    records = fetch_json(args.url)

    logging.info("Building DataFrame…")
    df = pd.json_normalize(records)
    df['description'] = df['description'].fillna('').astype(str)
    df['domain']      = df['domain'].fillna('').astype(str)

    logging.info("Computing features…")
    df = preprocess(df)

    logging.info("Computing recommendations…")
    df['recommendations'] = compute_recommendations(df, top_k=args.top_k)

    # 1) Full JSON: tutti i campi
    logging.info("Saving full JSON…")
    save_json(df, args.out_full)

    # 2) Summary JSON: solo campi chiave e top N per stelle
    summary_fields = [
        'full_name','description','stars','downloads',
        'watchers_count','development_status'
    ]
    logging.info("Saving summary JSON…")
    save_json(
        df,
        args.out_summary,
        fields=summary_fields,
        limit=args.summary_limit
    )

    logging.info("Done.")

if __name__=='__main__':
    main()
