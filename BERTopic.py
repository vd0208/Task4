import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

def compute_coherence(topic_model, docs, top_n=10):
    topic_words = []
    for t_id in sorted(topic_model.get_topics().keys()):
        if t_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(t_id)[:top_n]]
        if words:
            topic_words.append(words)

    tokenised  = [d.split() for d in docs]
    dictionary = Dictionary(tokenised)
    corpus     = [dictionary.doc2bow(t) for t in tokenised]

    cm = CoherenceModel(
        topics=topic_words,
        texts=tokenised,
        dictionary=dictionary,
        coherence='c_v'
    )
    return cm.get_coherence()


def compute_diversity(topic_model, top_n=10):
    all_words = []
    for t_id in sorted(topic_model.get_topics().keys()):
        if t_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(t_id)[:top_n]]
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


if __name__ == '__main__':

    SEED = 42
    np.random.seed(SEED)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('Set2')

    print('All imports OK.')

    # Load data  ← update path if needed
    DATA_PATH       = 'complete_preprocessing_final (1).csv'
    EMBEDDINGS_PATH = 'embeddings_minilm.npy'

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['trunc_processed_text']).reset_index(drop=True)
    df['trunc_processed_text'] = df['trunc_processed_text'].astype(str)

    print(f'Loaded {len(df):,} tickets')
    print(f"Ticket types: {df['type'].value_counts().to_dict()}")

    # Generate OR load sentence embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        # Fast path: load pre-computed embeddings saved by Embeddings.ipynb
        X_embeddings = np.load(EMBEDDINGS_PATH)
        print(f'Loaded embeddings from disk. Shape: {X_embeddings.shape}')
    else:
        # Slow path: generate fresh (takes ~3–5 min)
        print('Generating sentence embeddings with all-MiniLM-L6-v2 ...')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        X_embeddings = model.encode(
            df['trunc_processed_text'].tolist(),
            show_progress_bar=True,
            batch_size=64,
            convert_to_numpy=True
        )
        np.save(EMBEDDINGS_PATH, X_embeddings)
        print(f'Saved embeddings → {EMBEDDINGS_PATH}')

    print(f'Embedding shape: {X_embeddings.shape}')   # (27919, 384)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART 2 — NEW: BERTopic (AXIS2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # From here everything is new BERTopic code.

    N_TOPICS_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45]   # sweep as required by AXIS2 spec

    print('BERTopic imports OK.')
    print('Helper functions defined.')

    # CountVectorizer and ctfidf are stateless — safe to share across iterations
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        stop_words='english'
    )

    # c-TF-IDF: reduce over-representation of frequent words
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    print('Sub-components ready.')

    # Sweep over N_TOPICS_LIST

    docs = df['trunc_processed_text'].tolist()
    results = []        # collect metrics for comparison table
    fitted_models = {}  # store model objects for later inspection

    for n_topics in N_TOPICS_LIST:
        print(f'\n{"="*60}')
        print(f'  Fitting BERTopic  →  target topics = {n_topics}')
        print(f'{"="*60}')

        # ── Reset seed before each run for full reproducibility
        np.random.seed(SEED)

        # ── Recreate UMAP and HDBSCAN fresh each iteration
        umap_model = UMAP(
            n_neighbors=15,
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=SEED,
            low_memory=True
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            core_dist_n_jobs=1
        )

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            nr_topics=n_topics,
            top_n_words=10,
            calculate_probabilities=False,
            verbose=False
        )

        topics, _ = topic_model.fit_transform(docs, embeddings=X_embeddings)
        df[f'bertopic_{n_topics}'] = topics

        # ── Metrics ──────────────────────────────────────────────────
        n_found     = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers  = sum(1 for t in topics if t == -1)
        outlier_pct = n_outliers / len(topics) * 100

        # Silhouette (exclude outliers)
        mask = np.array(topics) != -1
        if mask.sum() > 1 and len(set(np.array(topics)[mask])) > 1:
            sil = silhouette_score(X_embeddings[mask], np.array(topics)[mask])
        else:
            sil = float('nan')

        # Coherence C_v
        coherence = compute_coherence(topic_model, docs)

        # Topic diversity
        diversity = compute_diversity(topic_model)

        # ARI / NMI vs ground-truth ticket type
        true_labels = pd.Categorical(df['type']).codes
        pred_no_out = np.array(topics)
        valid_mask  = pred_no_out != -1
        ari = adjusted_rand_score(true_labels[valid_mask], pred_no_out[valid_mask])
        nmi = normalized_mutual_info_score(true_labels[valid_mask], pred_no_out[valid_mask])

        print(f'  Topics found   : {n_found}')
        print(f'  Outliers       : {n_outliers} ({outlier_pct:.1f}%)')
        print(f'  Silhouette     : {sil:.4f}')
        print(f'  Coherence C_v  : {coherence:.4f}')
        print(f'  Diversity      : {diversity:.4f}')
        print(f'  ARI            : {ari:.4f}')
        print(f'  NMI            : {nmi:.4f}')

        results.append({
            'n_topics_target' : n_topics,
            'n_topics_found'  : n_found,
            'n_outliers'      : n_outliers,
            'outlier_pct'     : round(outlier_pct, 2),
            'silhouette'      : round(sil, 4),
            'coherence_cv'    : round(coherence, 4),
            'diversity'       : round(diversity, 4),
            'ari'             : round(ari, 4),
            'nmi'             : round(nmi, 4),
        })
        fitted_models[n_topics] = topic_model

    print('\n✓ Sweep complete.')

    # Summary comparison table
    results_df = pd.DataFrame(results)
    print('\nBERTopic Results Summary')
    print('='*80)
    print(results_df.to_string(index=False))

    # Save to CSV for report
    results_df.to_csv('bertopic_results_summary.csv', index=False)
    print('\nSaved: bertopic_results_summary.csv')

    # Pick the best model by Coherence C_v
    best_row    = results_df.loc[results_df['coherence_cv'].idxmax()]
    best_n      = int(best_row['n_topics_target'])
    best_model  = fitted_models[best_n]
    best_topics = df[f'bertopic_{best_n}'].tolist()

    print(f'Best model: n_topics_target = {best_n}')
    print(f'  Coherence C_v : {best_row["coherence_cv"]}')
    print(f'  Silhouette    : {best_row["silhouette"]}')
    print(f'  Diversity     : {best_row["diversity"]}')

    # Top words per topic (best model)
    print(f'\nTop 10 words per topic  (n_topics={best_n})')
    print('='*60)
    for t_id in sorted(best_model.get_topics().keys()):
        if t_id == -1:
            continue
        words = [w for w, _ in best_model.get_topic(t_id)[:10]]
        print(f'  Topic {t_id:>2}: {" | ".join(words)}')

    # Representative documents per topic
    print(f'\nRepresentative documents  (n_topics={best_n})')
    print('='*60)

    topic_arr = np.array(best_topics)
    for t_id in sorted(set(topic_arr)):
        if t_id == -1:
            continue
        mask    = topic_arr == t_id
        samples = df.loc[mask, 'trunc_processed_text'].head(2)
        print(f'\nTopic {t_id}:')
        for i, text in enumerate(samples, 1):
            print(f'  {i}. {text[:120]}...')

    # Visualisation 1: Topic size distribution
    topic_counts = pd.Series(best_topics).value_counts().sort_index()
    topic_counts = topic_counts[topic_counts.index != -1]   # exclude outliers

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(topic_counts.index.astype(str), topic_counts.values, color=sns.color_palette('Set2', len(topic_counts)))
    ax.set_title(f'BERTopic — Document Count per Topic  (target k={best_n})', fontsize=13)
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Number of documents')
    plt.tight_layout()
    plt.savefig('bertopic_topic_sizes.png', dpi=150)
    plt.show()
    print('Saved: bertopic_topic_sizes.png')

    # Visualisation 2: UMAP scatter (2D) coloured by BERTopic cluster
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0,
                   metric='cosine', random_state=SEED)
    X_2d = umap_2d.fit_transform(X_embeddings)

    topic_arr  = np.array(best_topics)
    valid_mask = topic_arr != -1
    palette    = sns.color_palette('tab10', len(set(topic_arr[valid_mask])))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Outliers in grey
    ax.scatter(X_2d[~valid_mask, 0], X_2d[~valid_mask, 1],
               s=3, alpha=0.2, color='lightgrey', label='Outlier (-1)')

    for i, t_id in enumerate(sorted(set(topic_arr[valid_mask]))):
        m = (topic_arr == t_id)
        ax.scatter(X_2d[m, 0], X_2d[m, 1],
                   s=5, alpha=0.5, color=palette[i], label=f'Topic {t_id}')

    ax.set_title(f'BERTopic — UMAP 2D Projection  (target k={best_n})', fontsize=13)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.legend(markerscale=4, loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig('bertopic_umap_scatter.png', dpi=150)
    plt.show()
    print('Saved: bertopic_umap_scatter.png')

    # Visualisation 3: Topic × ticket-type heatmap (alignment with ground truth)
    cross = pd.crosstab(
        df[f'bertopic_{best_n}'].replace(-1, 'outlier'),
        df['type']
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(cross)*0.5 + 1)))
    sns.heatmap(
        cross,
        annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, ax=ax
    )
    ax.set_title(f'BERTopic Topic × Ticket Type  (target k={best_n})', fontsize=13)
    ax.set_xlabel('Ticket Type')
    ax.set_ylabel('BERTopic Topic ID')
    plt.tight_layout()
    plt.savefig('bertopic_heatmap.png', dpi=150)
    plt.show()
    print('Saved: bertopic_heatmap.png')

    # Visualisation 4: Metrics across topic counts (sweep comparison)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    metrics = [
        ('coherence_cv', 'Coherence C_v', 'steelblue'),
        ('silhouette',   'Silhouette',    'darkorange'),
        ('diversity',    'Diversity',     'seagreen'),
    ]

    for ax, (col, label, color) in zip(axes, metrics):
        ax.plot(results_df['n_topics_target'], results_df[col],
                marker='o', color=color, linewidth=2)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Number of Topics (target)')
        ax.set_ylabel(label)
        ax.set_xticks(N_TOPICS_LIST)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('BERTopic — Metrics vs. Number of Topics', fontsize=13)
    plt.savefig('bertopic_metrics_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: bertopic_metrics_sweep.png')

    # Final printed summary (mirrors Embeddings.ipynb style)
    print('\n' + '='*70)
    print('BERTopic Analysis Complete  (AXIS2)')
    print('='*70)

    print(f'\nBest configuration:')
    for k, v in best_row.items():
        print(f'  {k:<20}: {v}')

    print('\nAll configurations:')
    print(results_df.to_string(index=False))
    print('\n' + '='*70)
