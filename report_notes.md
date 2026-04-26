# Report notes according to notebooks

### initial_clustering.ipynb **— Step 3: Initial Clustering
Purpose:** Initial exploration using TF-IDF only, to select the best k before Axis 1.
**Data:** 27,919 English customer support tickets, loaded from complete_preprocessing_final.csv, using the trunc_processed_text column (lemmatised, stopwords removed, truncated to 85 words).
**Representation:** TF-IDF vectoriser — 5,000 features, unigrams + bigrams (ngram_range=(1,2)), sublinear_tf=True, min_df=5, max_df=0.85.
**How k=5 was selected — three independent methods all agreed:**
1. **Elbow curve** (K-Means, k=3–25): inertia flattens after k=5; silhouette score peaks at k=5 (0.0381) and declines at higher k.
2. **HAC dendrogram** (Ward linkage, 500-ticket sample): natural cut at Ward distance ~2.0 yields 5 clean colour groups with a large gap above, confirming k=5.
3. **Multi-seed stability** (5 seeds × 4 k values): stability ARI = 0.9973 at k=5, drops sharply to 0.8899 at k=6, indicating the 6th cluster is not reproducible.

⠀**Multi-k × Multi-seed results table (K-Means on TF-IDF):**
| **k** | **Silhouette** | **ARI** | **NMI** | **Coherence** | **Stability** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 5 | 0.0381 | 0.2553 | 0.2583 | 0.7605 | 0.9973 |
| 6 | 0.0363 | 0.2856 | 0.3293 | 0.7623 | 0.8899 |
| 10 | 0.0348–0.0372 | 0.1606–0.1828 | 0.2968–0.3187 | 0.7230–0.7321 | 0.7621 |
| 15 | 0.0305–0.0351 | 0.1084–0.1351 | 0.2756–0.3026 | 0.7256–0.7602 | 0.8202 |
**K-Means vs HAC at k=5 (TF-IDF):**
| **Method** | **Silhouette** | **ARI** | **NMI** | **Coherence** |
|:-:|:-:|:-:|:-:|:-:|
| K-Means | 0.0381 | 0.2553 | 0.2583 | 0.7605 |
| HAC | 0.1055 | 0.2722 | 0.2925 | 0.7280 |
HAC achieves higher silhouette, ARI and NMI but lower coherence than K-Means on TF-IDF. This is because HAC on LSA-50 reduced space finds more geometrically separated clusters while K-Means on raw TF-IDF optimises for keyword-discriminative clusters.
**Five cluster themes identified (K-Means, k=5, seed=42):**
| **Cluster** | **Size** | **%** | **Theme** | **Top words** |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 5,514 | 19.7% | Healthcare & Security Issues | security, medical, data, breach, hospital |
| 1 | 12,292 | 44.0% | Technical Support & System Failures | issue, problem, resolve, restart, update |
| 2 | 3,754 | 13.4% | SaaS & Project Management | project, management, saas, integration |
| 3 | 3,763 | 13.5% | Digital Marketing & Brand Strategy | digital, brand, strategy, market, growth |
| 4 | 2,596 | 9.3% | Data Analytics & Investment | investment, analytics, optimize, decision |
**Plots produced:** elbow curve, silhouette vs k line, HAC dendrogram, K-Means scatter (LSA 2D), silhouette plot, top words bar charts, cluster size distribution bar chart, metrics heatmap (k × seed), metrics line plots across k.
**Key finding:** k=5 is optimal — confirmed by three independent methods. Cluster 1 (Technical Support) dominates at 44% of tickets, suggesting system failures are the most common customer issue. All 5 themes are clearly interpretable and distinct.

### Tfidf.ipynb **— TF-IDF Baseline (earlier version)
Purpose:** Earlier standalone TF-IDF notebook. Partially overlaps with initial_clustering.ipynb but contains the cross-tabulation of clusters vs ground-truth ticket types which is not in the other notebook.
**What ran:** K-Means (k=5), HAC (k=5), alignment with ground truth.
**K-Means results (k=5):** silhouette=0.0381, same cluster sizes and top words as initial_clustering — confirms consistency.
**HAC results (k=5, Ward, LSA-50):** silhouette=0.1055, ARI=0.2722, NMI=0.2925.
**Cross-tabulation (K-Means clusters vs ticket type — unique to this notebook):**
|  | **Change** | **Incident** | **Problem** | **Request** |
|:-:|:-:|:-:|:-:|:-:|
| Cluster 0 | 657 | 2,048 | 1,021 | 1,788 |
| Cluster 1 | 259 | 7,943 | 4,033 | 57 |
| Cluster 2 | 743 | 67 | 52 | 2,892 |
| Cluster 3 | 261 | 1,466 | 718 | 1,318 |
| Cluster 4 | 283 | 474 | 179 | 548 |
Cluster 1 is dominated by Incidents and Problems — confirming it captures genuine technical failure tickets. Cluster 2 is almost entirely Requests — confirming the SaaS/project management theme.
**Note for report:** This notebook's cross-tabulation is valuable evidence that clusters align with real ticket categories.

### Embeddings.ipynb **— Sentence Embeddings (separate exploration)
Purpose:** Early exploration of sentence embeddings before Axis 1. Results were later reused in axis1_representations.ipynb by saving embeddings to embeddings_minilm.npy.
**Model:** all-MiniLM-L6-v2 from sentence-transformers. Embeddings shape: (27,919 × 384). Saved to disk to avoid recomputing (~5–10 mins encoding time).
**K-Means (k=5, seed=42):**
* Silhouette: 0.1214, ARI: 0.1982, NMI: 0.2066, Coherence: 0.7304
* Cluster sizes: 5,985 / 3,999 / 3,910 / 3,861 / 10,164

⠀**HAC (k=5, Ward linkage, on raw embeddings):**
* Silhouette: 0.0970, ARI: 0.2354, NMI: 0.2519, Coherence: 0.7511
* Cluster sizes: 11,589 / 6,345 / 4,682 / 3,637 / 1,666

⠀**K-Means cluster themes (embeddings):**
* Cluster 0 (21.4%): Healthcare & Security — data, security, medical, breach
* Cluster 1 (14.3%): Data Analytics & Investment — investment, analytics, tool
* Cluster 2 (14.0%): Digital Marketing — digital, strategy, brand, market
* Cluster 3 (13.8%): SaaS & Project Management — integration, project, management
* Cluster 4 (36.4%): Technical Support — issue, problem, resolve, software

⠀**HAC cluster themes (embeddings):**
* Cluster 0 (41.5%): Technical Support (large dominant cluster)
* Cluster 1 (22.7%): Healthcare & Security
* Cluster 2 (16.8%): SaaS & Project Management
* Cluster 3 (13.0%): Digital Marketing
* Cluster 4 (6.0%): Data Analytics (smallest)

⠀**Key observation:** HAC on embeddings produces a more imbalanced distribution (41.5% in one cluster) than K-Means. The same 5 themes appear in both methods confirming genuine topic structure.

### axis1_representations.ipynb **— Axis 1: Text Representation Comparison
Purpose:** Full Axis 1 comparison of three representations: TF-IDF, Sentence Embeddings, Word2Vec. All use k=5, seed=42 (selected from initial_clustering).
**Hypothesis (stated before experiments):** Sentence Embeddings > Word2Vec > TF-IDF on coherence and interpretability, because dense representations capture semantic meaning beyond keyword overlap.
**Three representations:**
1. **TF-IDF** — same as initial_clustering (refit for consistency)
2. **Sentence Embeddings** — all-MiniLM-L6-v2, loaded from saved embeddings_minilm.npy (27,919 × 384)
3. **Word2Vec** — trained directly on the customer support corpus using gensim Word2Vec (vector_size=100, window=5, min_count=2, epochs=10). Vocabulary: 3,144 words. Document vectors = mean of word vectors. Saved to w2v_vectors.npy.

⠀**Main comparison table (K-Means, k=5, seed=42):**
| **Representation** | **Silhouette** | **ARI** | **NMI** | **Coherence** | **Stability** |
|:-:|:-:|:-:|:-:|:-:|:-:|
| TF-IDF | 0.0381 | 0.2553 | 0.2583 | **0.7605** | 0.9973 |
| Sentence Embeddings | 0.1214 | 0.1982 | 0.2066 | 0.7304 | **1.0000** |
| **Word2Vec** | **0.2595** | **0.3315** | **0.3766** | 0.7280 | 0.9996 |
**Multi-k comparison (all 3 representations at k=5,6,10,15):**
| **k** | **TF-IDF sil** | **Emb sil** | **W2V sil** | **TF-IDF ARI** | **Emb ARI** | **W2V ARI** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 5 | 0.0381 | 0.1214 | **0.2595** | 0.2553 | 0.1982 | **0.3315** |
| 6 | 0.0363 | 0.1070 | **0.1916** | 0.2856 | 0.1691 | **0.2481** |
| 10 | 0.0348 | 0.0860 | **0.1834** | 0.1828 | 0.1653 | **0.2007** |
| 15 | 0.0305 | 0.0802 | **0.1491** | 0.1351 | 0.1174 | **0.1455** |
Word2Vec leads silhouette at every k value. All three agree k=5 is optimal.
**Hypothesis outcome — PARTIALLY REJECTED:** Word2Vec outperformed Sentence Embeddings on silhouette, ARI and NMI. TF-IDF achieved highest coherence and near-perfect stability. The expected ranking (Embeddings > Word2Vec > TF-IDF) was not observed.
**Why Word2Vec outperformed Sentence Embeddings:** Word2Vec was trained directly on the customer support corpus, learning domain-specific vocabulary (medical-hospital-breach, analytics-investment-optimize). The sentence transformer was pre-trained on general text and applied zero-shot — it had no exposure to this specialised vocabulary. Domain adaptation explains the unexpected result.
**Why TF-IDF has highest coherence:** Coherence measures top-word co-occurrence within clusters. TF-IDF directly weights discriminative terms, naturally producing coherent word sets. Its near-perfect stability (0.9973) confirms keyword-based representations are robust to random initialisation.
**Qualitative consistency — all three representations identify the same 5 themes:**
* Healthcare & Security Issues (~20%)
* Technical Support & System Failures (~44%)
* SaaS & Project Management (~13%)
* Digital Marketing & Brand Strategy (~14%)
* Data Analytics & Investment (~9%)

⠀This cross-method agreement is itself a strong finding — the 5 issue categories genuinely exist in the data and are not artefacts of any single method.
**Plots produced:** comparison heatmap, metric bar charts, t-SNE (all 3), silhouette plots (all 3), top words bar charts (all 3), cluster size distribution (all 3), multi-k line plots.
