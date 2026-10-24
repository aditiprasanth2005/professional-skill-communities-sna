# Discovering Professional Skill Communities

**Graph-based analysis with Spectral Embedding + K-Means**

An interactive Streamlit research prototype that discovers groups of professional profiles from their skill lists. The app supports a real-world resume dataset and a synthetic fallback mode.

> **Data transparency:** the real-data mode uses the public Hugging Face `C0ldSmi1e/resume-dataset` (resume records with skill/category fields). It is **not a LinkedIn API export** and the project does not scrape LinkedIn. The graph represents similarity between skill profiles, not actual social connections.

## Features

- Fetches a public resume dataset on first run (no raw dataset is committed to this repository).
- Cleans comma/semicolon-separated skill phrases and creates a profile-skill matrix.
- Computes cosine similarity and builds a weighted k-nearest-neighbour graph.
- Runs Spectral Embedding followed by K-Means.
- Visualises the embedding and graph, summarises dominant skills per cluster, and explores individual records.
- Reports Silhouette, Davies–Bouldin and Calinski–Harabasz internal metrics.
- In real-data mode, reports ARI/NMI as exploratory comparisons with the dataset's category labels (not as accuracy or verified community ground truth).
- Exports a minimal results CSV without raw resume text, education or experience.
- Includes a synthetic fallback for offline demonstration.

## Run locally (macOS/Linux)

```bash
git clone <your-repository-url>
cd professional-skill-communities
a python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first real-data run downloads the dataset through Hugging Face and may take a few minutes. An internet connection is required. Choose **Synthetic demonstration** in the sidebar if the dataset cannot be downloaded.

## Dataset

- **Name:** Resume Dataset
- **Repository:** https://huggingface.co/datasets/C0ldSmi1e/resume-dataset
- **Publisher-reported licence:** Apache-2.0 (verify the current dataset card before reuse or redistribution)
- **Useful fields:** `id`, `category`, `skills`, `education`, `experience`, `text`, `html`
- **Fields used by this app:** record ID (for deduplication), category and skills. Raw text/HTML is not shown or exported.

The dataset card describes resume records and skills; it does not establish that the records are LinkedIn profiles. Accordingly, the project's real-data experiment is described as professional/resume skill community discovery rather than LinkedIn-network analysis.

## Method

1. Load real resume skills or generate synthetic examples.
2. Split and normalize skill phrases.
3. Build a binary profile-by-skill matrix (up to 400 frequent skill phrases).
4. Calculate cosine similarity.
5. Create an undirected weighted graph from each profile's strongest qualifying neighbours.
6. Compute a 2D spectral embedding from the graph affinity matrix.
7. Apply K-Means to the embedding.
8. Interpret clusters from dominant skills and evaluate with internal metrics.

### Metrics and interpretation

- **Silhouette:** higher generally indicates better separation.
- **Davies–Bouldin:** lower generally indicates tighter, better-separated clusters.
- **Calinski–Harabasz:** higher generally indicates better separation relative to within-cluster dispersion.
- **ARI/NMI:** compare predicted clusters with the dataset's category labels. These categories are imperfect proxy labels, not ground-truth network communities. They are not accuracy percentages.

Internal metrics are calculated in the two-dimensional embedding used by K-Means, so report this explicitly when presenting results.

## Responsible use and limitations

- Do not scrape LinkedIn or claim to have used LinkedIn data without authorised access and a documented data source.
- The graph edges indicate skill similarity, not friendship, endorsement or actual platform connections.
- Public resume datasets may still contain sensitive or identifying information. This app avoids displaying/exporting raw resume text and does not bundle the dataset. Review the dataset card and applicable licence before redistribution.
- Skill lists can be inconsistent, incomplete, noisy and uneven across occupations.
- Spectral embedding and K-Means are unsupervised; clusters need human interpretation.
- Dataset category labels are used only for exploratory comparison.

## Suggested GitHub structure

```text
professional-skill-communities/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── docs/
    └── methodology.md
```

## Team

- Aditi Prasanth
- Tanishq Tamarkar
- Abhiraj Deshmukh
- Yash Phogat

## Academic integrity

Report the dataset source, the exact sample size, parameters and metrics from your own run. Do not copy the example metrics from the synthetic prototype as real-data results. Clearly distinguish real dataset records from generated fallback examples.
