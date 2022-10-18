## How was this data collected?
See the [`compile_and_filter_dataset`](../compile_and_filter_dataset/) directory for step-by-step information on how I obtained and cleaned this data.

### Contents:

* **movies_cleaned.csv**
    * Movie title
    * Reddit post title
    * Reddit post ID
    * Reddit post URL
    * Over 900 rows

* **movies_comments.json**
    * Dictionary with Reddit post IDs as keys
    * Movie title
    * List of top 100 comments from post
    * UTC date of Reddit post

* **reddit_movies_final.csv**
    * Duplicate movie entries from `movies_cleaned.csv` have been removed
    * Reddit post ID
    * Movie title
    * Post year
    * Post month
    * Post day
    * Over 900 rows

* **comments_exploded.csv**
    * Same as `reddit_movies_final.csv`, but includes comments.
    * Each comment is its own row.
    * Over 70,000 rows

* **data_final.csv**
    * Similar to `comments_exploded.csv`, but matched with each movie's IMDb score
    * Some movie titles from Reddit are cleaned up more to match IMDb title