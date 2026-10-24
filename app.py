import re
import html
import warnings
from collections import Counter

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score

st.set_page_config(page_title="Professional Skill Communities", page_icon="🕸️", layout="wide")

DATASET_ID = "C0ldSmi1e/resume-dataset"
DATASET_URL = "https://huggingface.co/datasets/C0ldSmi1e/resume-dataset"

st.title("Discovering Professional Skill Communities")
st.caption("Graph-based community discovery · Spectral Embedding + K-Means")

with st.sidebar:
    st.header("Experiment settings")
    data_mode = st.radio("Data source", ["Real-world resume dataset", "Synthetic demonstration"], index=0)
    if data_mode == "Real-world resume dataset":
        n_profiles = st.slider("Profiles to analyse", 100, 600, 300, step=50)
    else:
        n_profiles = st.slider("Synthetic profiles", 40, 300, 160, step=20)
    k = st.slider("Number of clusters (K)", 2, 10, 4)
    similarity_threshold = st.slider("Minimum edge similarity", 0.05, 0.70, 0.15, step=0.05)
    neighbours = st.slider("Maximum neighbours per profile", 3, 10, 5)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    run_button = st.button("Run analysis", type="primary", use_container_width=True)

st.info(
    "Real-world mode uses a public resume dataset with skills, education, experience and job-category fields. "
    "It is **not a LinkedIn API export**, and the project does not scrape LinkedIn. Raw resume text is not displayed or exported."
)

SYNTHETIC_GROUPS = {
    "AI & Data": ["Python", "Machine Learning", "Deep Learning", "NLP", "SQL", "Data Analysis", "PyTorch"],
    "Cloud & DevOps": ["AWS", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD", "Linux"],
    "Software Engineering": ["Java", "C++", "Git", "REST APIs", "Testing", "System Design", "TypeScript"],
    "Product & UX": ["Product Management", "User Research", "Figma", "Prototyping", "Agile", "Wireframing", "Analytics"],
}

@st.cache_data(show_spinner=False)
def load_public_resume_data():
    """Load the public Hugging Face dataset; cache only structured fields used by the app."""
    from datasets import load_dataset
    dataset = load_dataset(DATASET_ID)
    frames = []
    for split_name in dataset.keys():
        frame = dataset[split_name].to_pandas()
        frame["_source_split"] = split_name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    # Keep only the fields needed for this project; do not propagate raw resume/HTML text.
    expected = [c for c in ["id", "category", "skills", "education", "experience"] if c in df.columns]
    df = df[expected].copy()
    df = df.drop_duplicates(subset=["id"] if "id" in df.columns else None)
    df["skills"] = df.get("skills", pd.Series("", index=df.index)).fillna("").astype(str)
    df["category"] = df.get("category", pd.Series("Unknown", index=df.index)).fillna("Unknown").astype(str)
    return df


def split_skills(value):
    if not isinstance(value, str):
        return []
    parts = re.split(r"[,;|\n\r]+", value)
    cleaned = []
    seen = set()
    for part in parts:
        skill = re.sub(r"\s+", " ", html.unescape(part)).strip(" \t:-•")
        skill = skill[:80]
        key = skill.casefold()
        if len(skill) >= 2 and key not in seen:
            seen.add(key)
            cleaned.append(skill)
    return cleaned


def make_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    group_names = list(SYNTHETIC_GROUPS)
    rows = []
    for i in range(n):
        primary = group_names[i % len(group_names)]
        chosen = []
        for group, skills in SYNTHETIC_GROUPS.items():
            p = 0.72 if group == primary else 0.08
            chosen.extend(s for s in skills if rng.random() < p)
        if not chosen:
            chosen = [rng.choice(SYNTHETIC_GROUPS[primary])]
        rows.append({"Profile_ID": f"SYN-{i+1:04d}", "Category": primary, "Skills_List": chosen})
    return pd.DataFrame(rows)


def prepare_profiles(mode, n, seed):
    if mode == "Synthetic demonstration":
        df = make_synthetic(n, seed)
        return df, True, "Synthetic demonstration data"
    raw = load_public_resume_data()
    raw["Skills_List"] = raw["skills"].map(split_skills)
    raw = raw[raw["Skills_List"].map(len) > 0].copy()
    if len(raw) < 10:
        raise ValueError("The selected dataset did not contain enough non-empty skills records.")
    raw = raw.sample(n=min(n, len(raw)), random_state=int(seed)).reset_index(drop=True)
    raw["Profile_ID"] = [f"RES-{i+1:04d}" for i in range(len(raw))]
    raw["Category"] = raw["category"].astype(str)
    return raw[["Profile_ID", "Category", "Skills_List"]], False, f"Hugging Face: {DATASET_ID}"


def run_pipeline(profile_df, k, threshold, neighbours, seed):
    # Limit vocabulary to frequent skill phrases to keep the experiment reproducible and tractable.
    counts = Counter(skill.casefold() for skills in profile_df["Skills_List"] for skill in skills)
    vocabulary = [s for s, _ in counts.most_common(400)]
    vocab_lookup = set(vocabulary)
    clean_lists = [[s for s in skills if s.casefold() in vocab_lookup] for skills in profile_df["Skills_List"]]
    # Preserve display forms while creating a normalized binary profile-skill matrix.
    display_by_key = {}
    for skills in profile_df["Skills_List"]:
        for skill in skills:
            display_by_key.setdefault(skill.casefold(), skill)
    features = [display_by_key[key] for key in vocabulary]
    matrix_rows = []
    for skills in clean_lists:
        keys = {s.casefold() for s in skills}
        matrix_rows.append([int(feature.casefold() in keys) for feature in features])
    X = pd.DataFrame(matrix_rows, columns=features, dtype=np.uint8)
    # Remove any zero-variance columns and empty rows.
    X = X.loc[:, X.sum(axis=0) > 0]
    if X.shape[1] < 2:
        raise ValueError("Not enough distinct skills were found. Try a larger dataset sample.")
    sim = cosine_similarity(X.to_numpy(dtype=float))
    np.fill_diagonal(sim, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(profile_df)))
    for i in range(len(profile_df)):
        candidates = np.argsort(sim[i])[::-1][:neighbours]
        for j in candidates:
            if i != j and sim[i, j] >= threshold:
                G.add_edge(i, int(j), weight=float(sim[i, j]))
    for i in range(len(profile_df)):
        if G.degree(i) == 0:
            j = int(np.argmax(sim[i]))
            if i != j and sim[i, j] > 0:
                G.add_edge(i, j, weight=float(sim[i, j]))
    A = nx.to_numpy_array(G, nodelist=range(len(profile_df)), weight="weight")
    # SpectralEmbedding works on the weighted graph affinity matrix.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=int(seed)).fit_transform(A)
    labels = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed)).fit_predict(embedding)
    out = profile_df.copy().reset_index(drop=True)
    out["Cluster"] = labels + 1
    out["Degree"] = [G.degree(i) for i in range(len(out))]
    out["Skills"] = ["; ".join(skills) for skills in out["Skills_List"]]
    out["Top-level category"] = out["Category"]
    sil = silhouette_score(embedding, labels) if len(set(labels)) > 1 and len(set(labels)) < len(labels) else np.nan
    dbi = davies_bouldin_score(embedding, labels) if len(set(labels)) > 1 else np.nan
    chi = calinski_harabasz_score(embedding, labels) if len(set(labels)) > 1 else np.nan
    categories = out["Category"].astype(str)
    ari = adjusted_rand_score(categories, labels) if categories.nunique() > 1 else np.nan
    nmi = normalized_mutual_info_score(categories, labels) if categories.nunique() > 1 else np.nan
    return out, X, sim, G, embedding, labels, {"silhouette": sil, "dbi": dbi, "chi": chi, "ari": ari, "nmi": nmi}

try:
    with st.spinner("Preparing profile data and discovering communities…"):
        profiles, is_synthetic, source_label = prepare_profiles(data_mode, int(n_profiles), int(random_seed))
        results, X, sim, G, embedding, labels, metrics = run_pipeline(profiles, int(k), float(similarity_threshold), int(neighbours), int(random_seed))
except Exception as exc:
    st.error("The analysis could not run.")
    st.exception(exc)
    st.markdown(f"**Dataset page:** {DATASET_URL}")
    st.markdown("Install the requirements, check your internet connection, and rerun. The real dataset is fetched from Hugging Face on first use.")
    st.stop()

# Header metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Profiles analysed", f"{len(results):,}")
m2.metric("Skill features", f"{X.shape[1]:,}")
m3.metric("Graph edges", f"{G.number_of_edges():,}")
m4.metric("Silhouette", f"{metrics['silhouette']:.3f}" if np.isfinite(metrics['silhouette']) else "N/A")
m5.metric("Clusters", str(k))

st.caption(f"**Data source:** {source_label} · **Mode:** {'Synthetic demonstration' if is_synthetic else 'Real-world resume skills'} · **Seed:** {random_seed}")

tab1, tab2, tab3, tab4 = st.tabs(["Community map", "Cluster interpretation", "Profile explorer", "Method & validation"])

with tab1:
    st.subheader("Professional skill communities")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=35, alpha=0.82)
    ax.set_xlabel("Spectral dimension 1")
    ax.set_ylabel("Spectral dimension 2")
    ax.set_title("2D spectral embedding coloured by K-Means cluster")
    ax.grid(alpha=0.18)
    st.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    pos = {i: embedding[i] for i in range(len(results))}
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.14, width=0.6, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=labels, cmap=plt.cm.Set2, node_size=38)
    ax2.set_title("Skill-similarity graph · each node represents one profile record")
    ax2.axis("off")
    st.pyplot(fig2)
    st.caption("Edges represent cosine similarity between extracted skill lists; they do not represent actual social connections between people.")

with tab2:
    st.subheader("What characterises each discovered cluster?")
    summary = []
    for cid in sorted(results["Cluster"].unique()):
        ix = results.index[results["Cluster"] == cid].tolist()
        skill_counts = X.iloc[ix].sum().sort_values(ascending=False)
        category_counts = results.iloc[ix]["Category"].value_counts()
        summary.append({
            "Cluster": int(cid),
            "Profiles": len(ix),
            "Top skills": ", ".join(skill_counts.head(8).index.tolist()),
            "Most frequent dataset category": category_counts.index[0] if len(category_counts) else "Unknown",
            "Category mix (top 3)": "; ".join(f"{name} ({count})" for name, count in category_counts.head(3).items()),
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    if not is_synthetic:
        st.warning("Dataset categories are provided labels for resumes, not verified ground-truth communities. Use them only as a comparison signal; interpret clusters primarily from the skills that dominate them.")
    else:
        st.warning("Synthetic groups are generated for demonstration and can make clusters artificially easy to separate.")
    st.subheader("Cluster size")
    size_df = results["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Profiles")
    st.bar_chart(size_df.set_index("Cluster"))

with tab3:
    st.subheader("Explore analysed records")
    selected = st.selectbox("Filter by cluster", ["All"] + [str(i) for i in sorted(results["Cluster"].unique())])
    filtered = results if selected == "All" else results[results["Cluster"] == int(selected)]
    st.dataframe(filtered[["Profile_ID", "Cluster", "Category", "Degree", "Skills"]], use_container_width=True, hide_index=True)
    export = results[["Profile_ID", "Cluster", "Category", "Degree", "Skills"]].copy()
    st.download_button("Download analysed skill records (CSV)", export.to_csv(index=False).encode("utf-8"), file_name="professional_skill_communities_results.csv", mime="text/csv")
    st.caption("Export includes only dataset record IDs, category, extracted skills and graph/clustering outputs. It excludes raw resume text, education and experience fields.")

with tab4:
    st.subheader("Methodology")
    st.markdown("""
    1. **Load data:** obtain the public resume-and-skills dataset (or generate synthetic profiles for a fallback demo).
    2. **Preprocess:** split and normalize comma/semicolon-separated skill phrases; remove empty skill records.
    3. **Vectorize:** create a binary profile-by-skill matrix using the most frequent skill phrases.
    4. **Measure similarity:** compute cosine similarity between profile skill vectors.
    5. **Build graph:** connect each profile to up to *k* similar profiles if similarity meets the threshold; link isolated nodes to their strongest non-zero match.
    6. **Spectral embedding:** transform the weighted graph affinity matrix into two dimensions.
    7. **Cluster:** apply K-Means to the embedding.
    8. **Interpret and evaluate:** inspect dominant skills, cluster sizes and internal metrics.
    """)
    st.markdown("**Internal metrics (computed on the 2D embedding):**")
    st.write(f"- Silhouette score: **{metrics['silhouette']:.3f}** (higher is generally better)." if np.isfinite(metrics['silhouette']) else "- Silhouette score: unavailable.")
    st.write(f"- Davies–Bouldin index: **{metrics['dbi']:.3f}** (lower is generally better)." if np.isfinite(metrics['dbi']) else "- Davies–Bouldin index: unavailable.")
    st.write(f"- Calinski–Harabasz index: **{metrics['chi']:.2f}** (higher is generally better)." if np.isfinite(metrics['chi']) else "- Calinski–Harabasz index: unavailable.")
    if not is_synthetic:
        st.markdown("**Exploratory comparison against the dataset's provided category labels:**")
        st.write(f"- Adjusted Rand Index (ARI): **{metrics['ari']:.3f}**")
        st.write(f"- Normalized Mutual Information (NMI): **{metrics['nmi']:.3f}**")
        st.caption("ARI/NMI compare cluster assignments with resume categories. These categories are not ground-truth graph communities; the scores are exploratory, not accuracy.")
    st.markdown("**Graph definition:** nodes are resume records; edges are weighted by cosine similarity between extracted skill lists. This is a skill-similarity graph, not a social-network connection graph.")
    st.markdown("**Reproducibility:** change the seed, sample size, K, threshold and neighbour count to observe how the communities change.")
    with st.expander("Dataset provenance and responsible use"):
        st.markdown(f"- Dataset: [{DATASET_ID}]({DATASET_URL})")
        st.markdown("- Dataset card lists an Apache-2.0 licence and fields for category, skills, education, experience and resume text.")
        st.markdown("- This app uses only the skills and broad category fields for analysis; it does not display or export raw resume/HTML text.")
        st.markdown("- The dataset is not claimed to be a LinkedIn dataset. Do not describe this experiment as LinkedIn analysis unless you obtain and document authorised LinkedIn data.")
        st.markdown("- Review the dataset card and licence before redistributing dataset records. This repository does not bundle the raw dataset.")

st.divider()
st.markdown("**Team:** Aditi Prasanth · Tanishq Tamarkar · Abhiraj Deshmukh · Yash Phogat")
