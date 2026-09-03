import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.set_page_config(page_title="Movie Search Engine", page_icon="🎬", layout="wide")

# ── Global settings ──────────────────────────────────────────
BM25_K1 = 1.5
BM25_B = 0.75
FIELD_WEIGHTS = {
    'title': 3,
    'overview': 1,
    'genres': 2,
    'keywords': 1.5,
    'cast': 1.5,
    'director': 2,
    'tagline': 1
}
FIELDS = list(FIELD_WEIGHTS.keys())

# ── Utility functions ────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def normalise(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return ' '.join(tokens)

def bm25_score(tf, idf_val, dl, avgdl, k1=BM25_K1, b=BM25_B):
    num = tf * (k1 + 1)
    den = tf + k1 * (1 - b + b * (dl / avgdl))
    return idf_val * (num / den)

def score_adjust(bm25_sc, vote_avg, vote_count, popularity, alpha=0.1):
    if bm25_sc <= 0:
        return 0.0
    vote_signal = (vote_avg / 10.0) if vote_avg else 0.5
    confidence = min(vote_count / 50, 1.0) if vote_count else 0.0
    pop_signal = min(math.log1p(popularity) / 10.0, 1.0) if popularity else 0.0
    metadata_signal = (vote_signal * confidence + pop_signal) / 2.0
    return bm25_sc * (1.0 + alpha * metadata_signal)

def clean_field(value):
    if pd.isna(value):
        return ''
    return str(value).strip()

# ── Load data and build index ────────────────────────────────
@st.cache_resource
def load_and_index():
    df = pd.read_csv('prepared_documents.csv')

    # ColBERT-specific document representation
    df['colbert_text'] = (
        'Title: ' + df['title_raw'].apply(clean_field) + '. ' +
        'Genres: ' + df['genres'].apply(clean_field) + '. ' +
        'Keywords: ' + df['keywords'].apply(clean_field) + '. ' +
        'Cast: ' + df['cast'].apply(clean_field) + '. ' +
        'Director: ' + df['director'].apply(clean_field) + '. ' +
        'Overview: ' + df['overview'].apply(clean_field)
    )

    N = len(df)

    # Baseline index on weighted_text
    inverted_index = defaultdict(list)
    tf_store = {}
    doc_lengths = {}

    for idx, row in df.iterrows():
        tokens = str(row['weighted_text']).split()
        tf_store[idx] = Counter(tokens)
        doc_lengths[idx] = len(tokens)
        for term in set(tokens):
            inverted_index[term].append(idx)

    avgdl = np.mean(list(doc_lengths.values()))

    idf_store = {}
    for term, doc_list in inverted_index.items():
        df_t = len(doc_list)
        idf_store[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    # Per-field indexes for linear and FreqComb
    field_tf = {f: {} for f in FIELDS}
    field_inv_idx = {f: defaultdict(list) for f in FIELDS}
    field_avgdl = {}
    field_idf = {f: {} for f in FIELDS}

    for field in FIELDS:
        lengths = []
        for idx, row in df.iterrows():
            tokens = str(row[field]).split() if pd.notna(row[field]) else []
            field_tf[field][idx] = Counter(tokens)
            lengths.append(len(tokens))
            for term in set(tokens):
                field_inv_idx[field][term].append(idx)
        field_avgdl[field] = np.mean(lengths) if lengths else 1.0
        for term, doc_list in field_inv_idx[field].items():
            df_t = len(doc_list)
            field_idf[field][term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    # Global IDF for FreqComb
    global_doc_freq = defaultdict(set)
    for f in FIELDS:
        for doc_id, counter in field_tf[f].items():
            for term in counter:
                global_doc_freq[term].add(doc_id)

    global_idf = {}
    for term, doc_set in global_doc_freq.items():
        df_t = len(doc_set)
        global_idf[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    weighted_doc_len = {
        idx: sum(FIELD_WEIGHTS[f] * sum(field_tf[f][idx].values()) for f in FIELDS)
        for idx in df.index
    }
    avg_weighted_dl = np.mean(list(weighted_doc_len.values()))

    # Overview-only index
    overview_tf = {}
    overview_inv = defaultdict(list)
    overview_lens = {}

    for idx, row in df.iterrows():
        tokens = str(row['overview']).split()
        overview_tf[idx] = Counter(tokens)
        overview_lens[idx] = len(tokens)
        for term in set(tokens):
            overview_inv[term].append(idx)

    overview_avgdl = np.mean(list(overview_lens.values()))
    overview_idf = {}
    for term, doc_list in overview_inv.items():
        df_t = len(doc_list)
        overview_idf[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    return (
        df, N,
        inverted_index, tf_store, doc_lengths, avgdl, idf_store,
        field_tf, field_inv_idx, field_avgdl, field_idf,
        global_doc_freq, global_idf, weighted_doc_len, avg_weighted_dl,
        overview_tf, overview_inv, overview_lens, overview_avgdl, overview_idf
    )

@st.cache_resource
def load_colbert():
    from ragatouille import RAGPretrainedModel
    return RAGPretrainedModel.from_pretrained('colbert-ir/colbertv2.0')

# ── Search functions ─────────────────────────────────────────
def search_overview(query, top_k=10):
    q_terms = normalise(query).split()
    if not q_terms:
        return []

    cands = set()
    for term in q_terms:
        cands.update(overview_inv.get(term, []))

    scores = []
    for doc_id in cands:
        score = sum(
            bm25_score(
                overview_tf[doc_id].get(t, 0),
                overview_idf.get(t, 0.0),
                overview_lens[doc_id],
                overview_avgdl
            )
            for t in q_terms if overview_tf[doc_id].get(t, 0) > 0
        )
        if score > 0:
            scores.append((int(doc_id), float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search_baseline(query, top_k=10):
    q_terms = normalise(query).split()
    if not q_terms:
        return []

    cands = set()
    for term in q_terms:
        cands.update(inverted_index.get(term, []))

    scores = []
    for doc_id in cands:
        score = sum(
            bm25_score(
                tf_store[doc_id].get(t, 0),
                idf_store.get(t, 0.0),
                doc_lengths[doc_id],
                avgdl
            )
            for t in q_terms if tf_store[doc_id].get(t, 0) > 0
        )
        if score > 0:
            scores.append((int(doc_id), float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search_linear(query, top_k=10):
    q_terms = normalise(query).split()
    if not q_terms:
        return []

    cands = set()
    for term in q_terms:
        for f in FIELDS:
            cands.update(field_inv_idx[f].get(term, []))

    scores = []
    for doc_id in cands:
        score = sum(
            FIELD_WEIGHTS[f] * bm25_score(
                field_tf[f][doc_id].get(t, 0),
                field_idf[f].get(t, 0.0),
                sum(field_tf[f][doc_id].values()),
                max(field_avgdl[f], 1.0)
            )
            for t in q_terms for f in FIELDS
            if field_tf[f][doc_id].get(t, 0) > 0
        )
        if score > 0:
            scores.append((int(doc_id), float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def search_freqcomb(query, top_k=10):
    q_terms = normalise(query).split()
    if not q_terms:
        return []

    cands = set()
    for term in q_terms:
        cands.update(global_doc_freq.get(term, set()))

    scores = []
    for doc_id in cands:
        score = 0.0
        dl = weighted_doc_len[doc_id]
        for term in q_terms:
            agg_tf = sum(FIELD_WEIGHTS[f] * field_tf[f][doc_id].get(term, 0) for f in FIELDS)
            if agg_tf == 0:
                continue
            score += bm25_score(agg_tf, global_idf.get(term, 0.0), dl, avg_weighted_dl)

        if score > 0:
            scores.append((int(doc_id), float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def rank_results(query, method='freqcomb', top_k=10):
    dispatch = {
        'overview': search_overview,
        'baseline': search_baseline,
        'linear': search_linear,
        'freqcomb': search_freqcomb,
    }

    raw = dispatch[method](query, top_k=top_k)
    output = []

    for doc_id, bm25_sc in raw:
        row = prepared_df.loc[doc_id]
        adj_score = score_adjust(
            bm25_sc,
            row['vote_average'],
            row['vote_count'],
            row['popularity']
        )
        ov = str(row.get('overview', ''))
        output.append({
            'rank': 0,
            'doc_id': int(doc_id),
            'title': row['title_raw'],
            'score': round(float(adj_score), 4),
            'overview_snippet': ov[:150] + '...' if len(ov) > 150 else ov,
        })

    output.sort(key=lambda x: x['score'], reverse=True)
    output = output[:top_k]

    for i, r in enumerate(output, 1):
        r['rank'] = i

    return output

# ── Load everything ──────────────────────────────────────────
(
    prepared_df, N,
    inverted_index, tf_store, doc_lengths, avgdl, idf_store,
    field_tf, field_inv_idx, field_avgdl, field_idf,
    global_doc_freq, global_idf, weighted_doc_len, avg_weighted_dl,
    overview_tf, overview_inv, overview_lens, overview_avgdl, overview_idf
) = load_and_index()

# ── UI ───────────────────────────────────────────────────────
st.title("🎬 Movie Search Engine")
st.caption("BM25 and ColBERT Information Retrieval")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    query = st.text_input(
        "Enter your query",
        placeholder="e.g. christopher nolan, survival stranded, superhero marvel"
    )

with col2:
    method = st.selectbox("Retrieval method", ['freqcomb', 'baseline', 'linear', 'overview'])

with col3:
    top_k = st.slider("Top K", min_value=5, max_value=20, value=10)

use_colbert = st.checkbox("Enable ColBERT re-ranking")

if query:
    with st.spinner('Searching...'):
        results = rank_results(query, method=method, top_k=top_k)

    if use_colbert:
        with st.spinner('Running ColBERT re-ranking...'):
            RAG = load_colbert()
            bm25_results = search_freqcomb(query, top_k=50)
            candidate_ids = [doc_id for doc_id, _ in bm25_results]
            candidate_docs = [prepared_df.loc[doc_id, 'colbert_text'] for doc_id in candidate_ids]

            reranked = RAG.rerank(
                query=query,
                documents=candidate_docs,
                k=min(top_k, len(candidate_docs))
            )

            content_to_ids = {}
            for doc_id, doc_text in zip(candidate_ids, candidate_docs):
                content_to_ids.setdefault(doc_text, []).append(doc_id)

            reranked_ids = []
            used_counts = {}

            for res in reranked:
                doc_text = res['content']
                used_counts.setdefault(doc_text, 0)
                if doc_text in content_to_ids and used_counts[doc_text] < len(content_to_ids[doc_text]):
                    reranked_ids.append(content_to_ids[doc_text][used_counts[doc_text]])
                    used_counts[doc_text] += 1

            results = []
            for rank, doc_id in enumerate(reranked_ids, 1):
                row = prepared_df.loc[doc_id]
                ov = str(row.get('overview', ''))
                results.append({
                    'rank': rank,
                    'doc_id': int(doc_id),
                    'title': row['title_raw'],
                    'score': None,
                    'overview_snippet': ov[:150] + '...' if len(ov) > 150 else ov,
                })

    st.markdown(f"### Results for: *{query}* — method: `{method}` {'+ ColBERT' if use_colbert else ''}")
    st.markdown(f"**{len(results)} results found**")

    for r in results:
        score_display = f"  —  score: {r['score']}" if r['score'] is not None else ''
        with st.expander(f"#{r['rank']}  {r['title']}{score_display}"):
            st.write(r['overview_snippet'])
