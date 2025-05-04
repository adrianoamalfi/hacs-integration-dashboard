import requests
import pandas as pd
import re
from datetime import timedelta

# 1. Download e normalizzazione JSON → DataFrame
url = "https://data-v2.hacs.xyz/integration/data.json"
resp = requests.get(url)
resp.raise_for_status()
raw = resp.json()
records = list(raw.values()) if isinstance(raw, dict) else raw
df = pd.json_normalize(records)

# 2. Colonne di base già viste: stars, recency_norm...
df['stars'] = df['stargazers_count'].fillna(0).astype(int)
max_stars = df['stars'].max() or 1
df['stars_norm'] = df['stars'] / max_stars

df['last_updated'] = pd.to_datetime(df['last_updated'], utc=True)
min_date = df['last_updated'].min()
max_date = df['last_updated'].max()
total_days = max((max_date - min_date).days, 1)
df['recency_norm'] = df['last_updated'].apply(lambda d: (d - min_date).days / total_days)

# 3. TOP DOWNLOADS
df['downloads'] = df['downloads'].fillna(0).astype(int)
max_dl = df['downloads'].max() or 1
df['downloads_norm'] = df['downloads'] / max_dl
# booleano: top 10% dei download
threshold = df['downloads'].quantile(0.90)
df['is_top_downloads'] = df['downloads'] >= threshold

# 4. STATO DI SVILUPPO basato su last_updated
now = pd.Timestamp.now(tz='UTC')
ages = (now - df['last_updated']).dt.days
def dev_status(days):
    if days > 365:
        return 'abandoned'
    if days > 180:
        return 'inactive'
    if days > 30:
        return 'maintained'
    return 'active'
df['development_status'] = ages.map(dev_status)

# 5. METRICHE AGGIUNTIVE
# 5.1 Open issues normalizzate
df['open_issues'] = df['open_issues'].fillna(0).astype(int)
max_issues = df['open_issues'].max() or 1
df['issues_norm'] = df['open_issues'] / max_issues

# 5.2 Popularity score (stelle + download)
df['popularity_score'] = 0.5 * df['stars_norm'] + 0.5 * df['downloads_norm']

# 5.3 Conteggio topics e lunghezza descrizione
df['topic_count'] = df['topics'].apply(lambda x: len(x) if isinstance(x, list) else (1 if isinstance(x, str) and x.strip() else 0))
df['description_length'] = df['description'].fillna('').apply(lambda s: len(s))

# 6. (Rimane uguale) Raccomandazioni basate soprattutto sui topics
def make_topics_set(row):
    t = row.get('topics')
    if isinstance(t, list):
        return set(t)
    elif isinstance(t, str) and t.strip():
        return {t.strip()}
    return set()

def make_semantic_set(row):
    tags = set()
    d = row.get('domain')
    if isinstance(d, str) and d.strip():
        tags.add(d.strip())
    words = re.findall(r'\w+', (row.get('description') or '').lower())
    tags |= {w for w in words if len(w) > 2}
    return tags

df['topics_set']   = df.apply(make_topics_set, axis=1)
df['semantic_set'] = df.apply(make_semantic_set, axis=1)

def jaccard(a, b):
    if not a and not b: return 0.0
    return len(a & b) / len(a | b)

# Pesi riequilibrati: ancora più peso ai topics
w_topics   = 0.6
w_semantic = 0.2
w_pop      = 0.15
w_recency  = 0.05

names       = df['full_name'].tolist()
topics_sets = df['topics_set'].tolist()
sem_sets    = df['semantic_set'].tolist()
pop_scores  = df['popularity_score'].tolist()
rec_n       = df['recency_norm'].tolist()

recommendations = []
for i in range(len(df)):
    scores = []
    for j in range(len(df)):
        if i == j: continue
        score = (
            w_topics   * jaccard(topics_sets[i], topics_sets[j]) +
            w_semantic * jaccard(sem_sets[i],    sem_sets[j])    +
            w_pop      * pop_scores[j]                             +
            w_recency  * rec_n[j]
        )
        scores.append((j, score))
    top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]
    recommendations.append([names[j] for j, _ in top5])

df['recommendations'] = recommendations

# 7. Controllo rapido
print(df[[
    'full_name','stars','downloads','is_top_downloads',
    'development_status','popularity_score','topic_count',
    'description_length','recommendations'
]].head(10))

import json


# 8. Esporta in JSON i record arricchiti senza NaN
#    Sostituisci NaN/NA con None → diventeranno `null` nel JSON
df_clean = df.where(pd.notnull(df), None)

output_records = df_clean.to_dict(orient='records')
with open('integrations_enhanced.json', 'w', encoding='utf-8') as f:
    json.dump(
        output_records,
        f,
        ensure_ascii=False,
        indent=2,
        default=str  # serve se hai ancora oggetti non serializzabili
    )

print(f"Salvato 'integrations_enhanced.json' con {len(output_records)} record (senza NaN).")