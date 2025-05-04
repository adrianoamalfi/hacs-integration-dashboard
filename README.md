# hacs-integration-dashboard

[![Daily Update](https://github.com/adrianoamalfi/hacs-integration-dashboard/actions/workflows/daily_update.yml/badge.svg)](https://github.com/adrianoamalfi/hacs-integration-dashboard/actions)
[![GitHub Pages](https://github.com/adrianoamalfi/hacs-integration-dashboard/workflows/GitHub%20Pages/badge.svg)](https://adrianoamalfi.github.io/hacs-integration-dashboard/)

**HACS Integration Dashboard** è un sito statico e un dataset JSON aggiornato quotidianamente che aggrega le integrazioni Home Assistant personalizzate con metadati arricchiti e raccomandazioni.

## Funzionalità

* Download giornaliero dei dati HACS da `https://data-v2.hacs.xyz/integration/data.json`
* Arricchimento con:

  * Conteggio stelle e popolarità normalizzata
  * Conteggio download e flag per top-downloads
  * Stato di sviluppo (active, maintained, inactive, abandoned)
  * Raccomandazioni avanzate basate su topics, descrizione, domain, popolarità e recency
  * Metriche aggiuntive: issue aperte, numero di topics, lunghezza descrizione
* Generazione automatica di `integrations_enhanced.json` nella cartella `docs/`
* Hosting tramite GitHub Pages

## Struttura del repository

```
.
├── .github
│   └── workflows
│       └── daily_update.yml        # Workflow GitHub Actions
├── docs
│   ├── index.html                 # Entry point del sito statico
│   └── integrations_enhanced.json # Output JSON arricchito
├── update_json.py                 # Script di aggiornamento
├── requirements.txt               # Dipendenze Python
└── README.md                      # Questo file
```

## Guida rapida

### Prerequisiti

* Python 3.7+
* pip

### Installazione

```bash
git clone https://github.com/adrianoamalfi/hacs-integration-dashboard.git
cd hacs-integration-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Aggiornamento manuale

```bash
python update_json.py
```

Il comando genera il file `docs/integrations_enhanced.json`.

### GitHub Actions

Il workflow `.github/workflows/daily_update.yml` esegue ogni giorno a mezzanotte UTC (e su dispatch manuale):

1. Download dei dati HACS
2. Arricchimento del dataset
3. Commit & push del JSON in `docs/`
4. Deploy su GitHub Pages

Assicurati di abilitare le **Actions** con permessi di lettura e scrittura, o di configurare un Personal Access Token (`PAT_HACS`) come secret.

### Accesso al sito

Visita:
[https://adrianoamalfi.github.io/hacs-integration-dashboard/](https://adrianoamalfi.github.io/hacs-integration-dashboard/)

## Personalizzazione

* Modifica i pesi delle raccomandazioni in `update_json.py` (variabili `w_topics`, `w_semantic`, `w_pop`, `w_recency`).
* Regola soglie per lo stato di sviluppo o percentili download nel script.
* Sostituisci `index.html` con un frontend dinamico (React, Vue, ecc.) per consumare il JSON.

## License

Questo progetto è rilasciato sotto licenza MIT. Aggiungi un file `LICENSE` se desideri includere il testo completo.
