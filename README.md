# Movie Search Engine: BM25 and ColBERT

An information-retrieval system for searching 4,799 movies using structured TMDB metadata. The project compares classical BM25 retrieval strategies with a ColBERT v2 neural re-ranker and includes a Streamlit search interface.

This was developed as a group coursework project for **ECS736P Information Retrieval** in the MSc Data Science and Artificial Intelligence programme.

**Authors:** Aslı Akel, Ayan Maharramli, Mariya Mahmood, and Taslin Osman.

## Project highlights

- Builds document representations from movie titles, plot summaries, genres, keywords, cast, director, and taglines.
- Implements an inverted index, BM25 scoring, field-weighted retrieval, and Robertson et al.'s FreqComb approach from first principles.
- Uses ColBERT v2 to re-rank the top 50 BM25 candidates through contextual late interaction.
- Evaluates five retrieval configurations across 30 manually judged queries using P@5, P@10, Recall@10, F1@10, MAP, and NDCG@10.
- Includes query-type analysis, Wilcoxon signed-rank tests, field-weight sensitivity analysis, and explicit failure-mode investigation.
- Provides both reproducible notebooks and an interactive Streamlit application.

## Selected results

The figures below are taken directly from the saved notebook outputs and represent mean performance across the same 30-query evaluation set.

| Retrieval configuration | P@5 | P@10 | Recall@10 | F1@10 | MAP | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Overview-only BM25 | 0.3200 | 0.2533 | 0.1788 | 0.2080 | 0.1259 | 0.2432 |
| Weighted-text BM25 | 0.6400 | 0.5833 | 0.4177 | 0.4805 | 0.3435 | **0.5382** |
| Linear field-score combination | 0.5067 | 0.4733 | 0.3440 | 0.3929 | 0.2460 | 0.4158 |
| FreqComb | **0.6800** | 0.5900 | 0.4211 | 0.4851 | **0.3527** | 0.5372 |
| ColBERT re-ranker | 0.6333 | **0.5933** | **0.4262** | **0.4900** | 0.3470 | 0.5338 |

FreqComb achieved the strongest P@5 and MAP, while ColBERT produced the best P@10, Recall@10, and F1@10. Weighted-text BM25 achieved the highest NDCG@10 by a narrow margin. The ColBERT-versus-FreqComb NDCG@10 difference was not statistically significant in the 30-query evaluation (two-sided Wilcoxon test, *p* = 0.3581), so the results are presented as a comparative analysis rather than evidence of universal superiority.

## Repository structure

```text
.
├── app/
│   └── app.py
├── data/
│   └── README.md
├── notebooks/
│   ├── movie_search_engine_pipeline.ipynb
│   ├── movie_search_engine_pipeline.py
│   └── demo_movie_search_engine.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

The pipeline notebook contains data preparation, indexing, retrieval, evaluation, statistical testing, and ColBERT re-ranking. The demo notebook loads the accompanying Python pipeline and presents compact interactive examples. The Streamlit app provides a browser-based search interface.

## Data access

The CSV files are intentionally not included. Download the **TMDB 5000 Movie Dataset** from Kaggle and follow [the data setup instructions](data/README.md):

<https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>

The pipeline expects `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and generates `prepared_documents.csv`. The generated file is also excluded because it contains transformed versions of the source movie metadata.

TMDB states that its API/data may be used for non-commercial purposes with attribution. This project is educational and is not endorsed or certified by TMDB.

## Running the project

Python 3.10 is recommended because the original coursework environment used Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then:

1. Download the two source CSV files and place them beside the pipeline notebook.
2. Open `notebooks/movie_search_engine_pipeline.ipynb` and run it from top to bottom. Restart the runtime after the dependency-installation cell if prompted.
3. The pipeline creates `prepared_documents.csv` and retains the saved aggregate evaluation outputs.
4. To use the compact demo, keep `movie_search_engine_pipeline.py` beside `demo_movie_search_engine.ipynb` and run the demo notebook.
5. To run the web interface, place `prepared_documents.csv` beside `app/app.py` and execute:

```bash
streamlit run app/app.py
```

ColBERT downloads `colbert-ir/colbertv2.0` on first use. GPU acceleration is recommended but the saved run demonstrates that execution can fall back to CPU. Set `RUN_COLBERT = False` in the pipeline when only the BM25 experiments are required.

## Responsible use and limitations

- The relevance judgements cover 30 coursework queries and should not be interpreted as a universal benchmark of movie-search quality.
- Some judgements are subjective; the notebook exposes the query set and graded relevance labels for inspection.
- Exact-match lexical retrieval remains vulnerable to vocabulary mismatch, which is analysed explicitly in the notebook.
- Popularity and rating metadata provide a small score adjustment and may encode popularity bias.
- No Kaggle/TMDB CSV data, credentials, tokens, or model weights are committed.

## Data attribution

- Kaggle dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Data source: [The Movie Database (TMDB)](https://www.themoviedb.org/)
- TMDB usage guidance: [TMDB API FAQ](https://developer.themoviedb.org/docs/faq)

This product uses the TMDB API/data but is not endorsed or certified by TMDB.

