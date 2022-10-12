## Data Collection

The dataset used in this notebook was compiled over the several Jupyter Notebooks in this directory. You can browse the notebooks for the detailed process. Here is a summary of the work done to collect the data.

### 1. Get movie discussions
Notebook: [1_getting_movie_discussions.ipynb](1_getting_movie_discussions.ipynb)

Use the PRAW library, a wrapper for the Reddit API.
* Collect the URLs and titles for all available official movie discussions on Reddit's r/movies. This should be 1,000 rows. 
    * Filter out discussions not related to movies, such as discussions of award ceremonies and end-of-year threads.
    * Use RegEx to clean up movie titles. Strip words from the post's title like
        * "official discussion"
        * "spoilers"
        * "international release"
    *  Save resulting dataframe as `movies_cleaned.csv`.

### 2. Get comments from movies
Notebook: [2_getting_comments_json.ipynb](2_getting_comments_json.ipynb)

Using PRAW and the URLS collected in the last notebook:
* Get the top 100 comments from each discussion thread (some will have fewer than 100) as well as the date the thread was posted.
    * The date will be useful when trying to match the Reddit post with the movie's IMDb score.
    * Save comments and dates in a dictionary format with the discussion's Reddit ID as the key.
    * This dictionary should be read as a dataframe and transposed.
* Save the resulting dictionary as a JSON file `movies_comments.json`.

### 3. Clean comments
Notebook: [3_cleaning_comments.ipynb](3_cleaning_comments.ipynb)

Using the previously saved JSON file:
* Read it in as a dataframe.
* Combine the comments from movies that were discussed on Reddit twice. Some movies had an discussions for international releases and US releases. Choose one post and ID to act as the ID for this movie.
    * This requires some granular work, since some movies simply have the same title, etc.
* Save this dataframe as a CSV, excluding the comments, as `reddit_movies_final.csv`.
* Explode comments. Resulting dataframe should have tens of thousands of rows.
* Drop duplicate comments.
    * More granular work here. Not all duplicates should be dropped. Only ones that appear to be spam or administrative.
* Remove administrative comments from each discussion. They only show up in a few discussions, but they are usually the first comment and contain keywords like "r/movies", "pinned", or "FYI".
* Save the resulting dataframe as `comments_exploded.csv`.

### 4. Merge movies with IMDb score

Notebook: [4_merge_reddit_imdb.ipynb](4_merge_reddit_imdb.ipynb)
* Download IMDb datasets "title.basics.tsv.gz" and "title.ratings.tsv.gz" from https://www.imdb.com/interfaces/
* Read in the datasets as dataframes. Drop non-movies and merge them on the unique identifier.
* Merge this IMDb dataset with the Reddit movies dataset. Use movie title and release date as the features to merge on.
    * Filter out duplicates. A surprising number of blockbuster movies share a release year and title with other, lesser known movies.
        * This requires granular work and care. Some Reddit movies might match with the wrong IMDb movie. Some Reddit movies were discussed in the year after the movie was released.
    * Some movies in the Reddit dataframe need their titles changed to match the IMDb counterpart (misspelled).
* Merge the resulting dataset with the comments dataset.
* Save resulting data as `data_final.csv`.