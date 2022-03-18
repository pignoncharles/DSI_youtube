# Charles Pignon 2022 Capstone

## Finding the most polarizing words from Youtube videos

This project uses machine learning and NLP to identify the most polarizing words from Youtube videos. The source is a kaggle database of 6351 unique Youtube videos that were trending in the US between December 2017 and May 2018.

Data processing is performed in the "Youtube project.ipynb" notebook. Visualization via Heroku app is performed using the "app.py" script.

Steps:
1. Data is read in and pre-processed: the sum of likes and dislikes are calculated per video, then used to calculate the like/dislike ratio per video. This metric has a heavily skewed distribution, therefore it is log-transformed to improve model performance.
2. Data is read into a ML pipe using the sklearn framework. Text in video titles and tags is tokenized using a CountVectorizer, excluding common stop words such as "the". An random forest model is used to predict log(like/dislike ratio) from title & tags text. The model was optimized using hyperparameter tuning.
3. Model outputs are stored and used for visualization on a Heroku app. Although the model could be used to predict like/dislike ratio from other videos, we will use it to assess which words in the video titles and tags were the most polarizing. To do this, we extract the feature importances from random forest. The features (i.e. words) with the most importance are those which the model selects as having the most influence on predicting the like/dislike ratio, therefore we can assess them as the most polarizing.
4. The Heroku app allows interactive exploration of these results. The app displays model performance, and allows users to explore the most polarizing words from the dataset. Some interesting results: topical subjects such as "trump" and "news" apear in the top 5 most polarizing words, reflecting the engagement towards politics and news outlets of the later 2010s. "funny" is ranked n. 7, and videos mentioning this word tend to have above-average log(like/dislike) ratio (Fig. 2). Surprisingly, the term "bowl" is ranked 9th most polarizing - upon closer inspection of these videos (Table 2), most of them refer to the February 2018 Super Bowl.

Source:
https://www.kaggle.com/datasnaek/youtube-new?select=CAvideos.csv
