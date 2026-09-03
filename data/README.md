# Dataset setup

The dataset is not redistributed in this repository.

Download **TMDB 5000 Movie Dataset** from Kaggle:

<https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata>

After reviewing and accepting the source terms, place these files in the working directory used to run the pipeline notebook:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The pipeline merges and preprocesses the source files and generates:

```text
prepared_documents.csv
```

The Streamlit application reads `prepared_documents.csv`. None of these CSV files should be committed to this repository.

This product uses the TMDB API/data but is not endorsed or certified by TMDB. Follow the attribution and usage requirements published by TMDB and the dataset provider.

