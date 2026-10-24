# Methodology notes

## Research question
Can professional skill communities be discovered by representing resume skill lists as a similarity graph and clustering a spectral embedding?

## Representation
Each record is a node. Skill phrases are normalized and converted into a binary feature vector. Cosine similarity between two vectors is used as edge weight.

## Graph construction
For each node, consider its top `neighbours` most similar records. Add an undirected edge if the similarity meets the configured threshold. An isolated node is linked to its strongest non-zero match where possible.

## Clustering pipeline
A weighted adjacency/affinity matrix is created from the graph. `sklearn.manifold.SpectralEmbedding` maps it to two dimensions, then K-Means assigns clusters. Cluster labels are arbitrary identifiers; interpret them using dominant skills.

## Evaluation
Silhouette, Davies–Bouldin and Calinski–Harabasz scores are computed on the 2D embedding. In real-data mode, ARI and NMI compare clusters to the dataset's provided occupation category. The category is not a verified community label, so external scores are exploratory only.

## Data statement
The app uses `C0ldSmi1e/resume-dataset` from Hugging Face in real-data mode. It does not assert the records are LinkedIn profiles. The raw dataset is fetched at runtime and is not stored in this repository. Only skills and broad categories are used in the visible app and export.
