**README is a work in progress**
# **NLP Movie Rater**

## Using Neural Networks and Reddit Comments to Predict Movie Ratings

<img src="./images/moshed_popcorn.jpg" alt="popcorn and movie tickets">

Flatiron School Data Science: Capstone Project

- **Author**: Zaid Shoorbajee
- **Instructor**: Morgan Jones
- **Pace**: Flex, 40 weeks
# Premise

This project aims to predict the IMDb rating of movies using comments from Reddit. The point is to help a (fictitious) movie studio anticipate the reception of its movies before release.

The final model chosen for this task is a deep neural network. It takes in text data that has pre-processed with natural language processing methods and outputs an IMDb score.

The final model was able to predict scores with an average error (RMSE) of 0.57 points on IMDb's 1–10 scale.

For the full 
## Business Understanding <a name="biz"></a>

A movie studio, Cinedistance, wants to know how well its movies will be received by audiences before it releases them. After all the hard work of producing a big-budget movie, there are still many [decisions about the marketing and release of the movie](https://www.nfi.edu/film-marketing/). How much should the studio invest in marketing? Should it release it to a [streaming or theaters](https://www.hollywoodreporter.com/tv/tv-news/netflix-forgoes-wide-release-martin-scorseses-irishman-1234382/)? To make these decisions, it's helpful to know how much audiences will actually like the movie first.

Cinedistance wants to get qualitative comments from average moviegoers *before* a movie's official release. It is also employing a data scientist to use these comments to predict a rating for the movie, like an IMDb score. It's piloting a program where the movie is released to a focus group of about 100 average moviegoers. These viewers are asked to sign an NDA, watch the movie, and submit their thoughts about the movie as if they are commenting on an internet comment section.

The task of the data scientist is to apply natural language processing (NLP) and machine learning (ML) to the focus group's comments in order to predict a movie's IMDb score. These qualitative comments and predicted score can inform Cinedistance's decision-making about the movie's areas of improvement, marketing, and distribution. The studio may decide to do re-shoots, re-edits, or even [kill the movie](https://variety.com/2022/film/news/batgirl-not-released-warner-bros-hbo-max-1235331897/).
## Data Understanding <a name="data_understanding"></a>

### Dataset <a name="dataset"></a>

Movie fans do do the type of commentary that Cinedistance is looking for everyday in places like the [Reddit community r/movies](reddit.com/r/movies). On **r/movies**, Reddit users share news and opinions about movies. Additionally, for most major movies that come out, the subreddit hosts an official discussion of the movie. These official discussions contains text data that can be the basis to train an ML model to predict IMDb scores.

Reddit has an API that allows developers to scrape such information. The [PRAW library](https://praw.readthedocs.io/) simplifies this, acting as a wrapper for the API. Using PRAW, [I scraped](./compile_and_filter_dataset/) from r/movies the **highest-voted 100 comments of as many official movie discussions still indexed on Reddit**. Some discussions had fewer than 100 comments. I also downloaded [freely available ratings  data from IMDb](https://www.imdb.com/interfaces/) and matched scores to r/movies discussions.

 The resulting dataset contains:
* 70,693 comments from 922 movies **(main features)**
* Movie title
* Reddit post ID and IMDb ID
* Reddit discussion date
* IMDb average rating **(target variable)**
* Number of IMDb votes
* Movie runtimes
* Genres

### NLP <a name="nlp"></a>

The core type of data being used for this task is the text of Reddit comments. This is **unstructured data** and requires natural language processing (NLP) techniques in order to be interpretable by a machine learning model, such as a deep neural network. 

Working with natural language is messy; different comments can have many of the same words, but context changes everything. It's easy for people to discern the difference, but for a computer, it's not so simple. To make comments interpretable by a neural network, this project uses the following NLP techniques:

* Tokenization
* Lemmatization
* Removing stop words
* TF-IDF Vectorization
* Part-of-speech tagging
* Sentiment analysis
* Meta-feature extraction

The idea is that converting comments into the signals listed above should help a machine learning model to discern a relationship between the comments and IMDb scores using hidden patterns.
## Data Analysis and Feature Engineering
TK TK TK
## Modeling <a name="modeling"></a>

This project ultimately is a regression task. I will be using [TensorFlow through the Keras interface](https://www.tensorflow.org/api_docs/python/tf/keras) in order to build a deep neural network. The neural network will be trained on a preprocessed version of the r/movies dataset that I have built.
## Scoring and Evaluation <a name="score"></a>

The business case that a movie studio wants to know the IMDb rating of a move before it is released. The IMDb scores the models are trained on have a degree of variability that can't be accounted for 100% of the time with predictions. The R-squared score tells us what percent of the variability of the target variable the model accounts for, so R-squared will be reported to the stakeholder.

Models in this notebook will also report loss as mean squared error (MSE) and root mean squared error (RMSE), which are a measures of average distance of predictions from the true target values. Minimizing MSE is what the models will be optimizing for.
## Conclusion 


### Recommendations

As discussed earlier, the movie studio Cinedistance wants to use a focus group's comments to predict their movies' IMDb score before official release. With the final model in this project, the focus group's comments would need to be pre-processed in the same way and then fed into the model. 

The model's output is an IMDb score that is expected to be, on average, within 0.57 points of a given movie's future IMDb score if it were released in its current state. The studio can then use that score to inform its decision-making around the movie's marketing, release, and possible changes to the movie.

The studio also has qualitative feedback on its movie alongside the predicted score. If the score is particularly low, the studio can review that feedback that the focus group provided and identify potential improvements. This could lead to re-shoots or re-edits that would hopefully improve the movie. 

Bear in mind that the average IMDb score from the dataset is about 6.5. It's up to the studio how much higher than this score it wants its movie to be before release — or how much lower it should be before taking drastic measures like scrapping the movie.

### Limitations

The models in this project are trained on 922 movies that were discussed on the Reddit community r/movies. This subreddit holds official discussions of many major releases, including big budget blockbusters, international movies, and art house movies. But this is still probably not a representative sample for all movies. Independent, low-budget films don't often get official discussions, for example. Thus, this model, will probably only produce accurate results for large, established movie studios. Additionally the r/movies subreddit has been around since 2008, but the final model is only trained on movies discussed between 2016 and 2022, since that was what was available to request through Reddit's API. A larger, more diverse sample of movie discussions to train on would make this model better-suited to rate more types of movies.

### Future Work

#### Deployment
Now that we have a final model, the next step would be to deploy it. The pre-processing steps would need to be deployed as well. 

This could be a Python file that:
* takes in a list of comments from Cinedistance's focus group about an unreleased movie.
* applies all of the pre-processing functions and transformations in this notebook.
* performs TF-IDF vectorization on the lemmatized discussions, using the vectorizer trained in this notebook.
* scales all features, using the standard scaler trained in this notebook.
* loads the final model from this notebook.
* returns a predicted IMDb score.

<!-- 
    This includes:* cleaning discussions* tokenizing and lemmatizing features* giving discussions a sentiment score* POS tagging* engineering meta-features -->

This Python app could also itelf be deployed as a web app to simplify access within the company.

<!-- * Transfer learning?
    * Trailers on YouTube
    * Recommender system -->

#### Use different data
As mentioned in "Limitations", the model is limited by the data that was available from r/movies. There are other websites that can provide a larger amount of similar data.

The website [Letterboxd](https://letterboxd.com/), for example, is a social media website dedicated to reviewing and discussing movies. It has a much thorougher catalog of movies, its own rating system, and an API. Using Letterboxd as an alternative to Reddit for this project could mean training a more robust model on thousands of movies, rather than just over 900.