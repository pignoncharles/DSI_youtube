import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title('Charles Pignon 2022 Capstone')
st.write("Finding the most polarizing words from Youtube videos")

#pull data
pred_vs_ground_truth = pd.read_csv('pred_vs_ground_truth.csv')
feature_importances = pd.read_csv('feature_importances.csv')
like_dislike_ratio = pd.read_csv('like_dislike_ratio.csv')

#show model results
st.header('1. Model performance')
#scatter model performance
st.write('Model performance: R2 = '+str(pred_vs_ground_truth['r2'].unique()[0]))
fig_1 = plt.figure(figsize=(10, 4))
sns.regplot(data = pred_vs_ground_truth,x='Log_like_dislike_ratio_ground_truth', y='Predicted_log_like_dislike_ratio', ci=None)
st.pyplot(fig_1)
st.write('Figure 1: Model performance chart: predicted vs. actual log(like/dislike ratio)')
# st.write('test numpy: '+str(np.log(.1)))

#print top n features
n_features_displayed = st.sidebar.selectbox(
    'N. top words to display',
[5,10,25,50])
st.write(feature_importances[['feature_names','importances']].head(n_features_displayed))
st.write('Table 1: top '+str(n_features_displayed)+' most polarizing words')

#adjust based on word selection
st.header('2. Explore polarizing words')
top_words=list(set(feature_importances.head(n_features_displayed)['feature_names']))
top_words.sort()
selected_top_word = st.sidebar.selectbox(
    'Select a word',
     top_words)
selected_top_word=str(selected_top_word)

#format to grab whole words
selected_top_word_formatted= r"\b({test_word_to_format})\b".format(test_word_to_format = selected_top_word)

'You selected: ', selected_top_word

#filter to word selection
test_plot = like_dislike_ratio.copy()
test_plot['contains_selected_word'] = test_plot['title_tags'].str.lower().str.contains(selected_top_word_formatted)

#show videos containing word
test_plot2 = test_plot[test_plot['contains_selected_word']]
test_plot3 = test_plot2[['title','tags','log_like_dislike_ratio','likes','dislikes','views']]
st.write(test_plot3)
st.write('Table 2: Videos mentioning '+selected_top_word)

#barplot selected vs other
fig_2 = plt.figure(figsize=(10, 4))
sns.barplot(data = test_plot,x='contains_selected_word', y='log_like_dislike_ratio')
st.pyplot(fig_2)
st.write('Figure 2: Mean + CI of log(like/dislike ratio) for videos mentioning '+selected_top_word+' vs. other videos')

# # scatter coloring by word
fig_3 = plt.figure(figsize=(10, 4))
sns.scatterplot(data = test_plot, x = 'dislikes', y = 'likes',hue = 'contains_selected_word')
st.pyplot(fig_3)
st.write('Figure 3: N. likes and dislike ratio for videos mentioning '+selected_top_word+' vs. other videos')

st. write('This project uses machine learning and NLP to identify the most polarizing words from Youtube videos. The source is a kaggle database of 6351 unique Youtube videos that were trending in the US between December 2017 and May 2018. Data processing is performed in the "Youtube project.ipynb" notebook. Visualization via Heroku app is performed using the "app.py" script.')

st. write('1. Data is read in and pre-processed: the sum of likes and dislikes are calculated per video, then used to calculate the like/dislike ratio per video. This metric has a heavily skewed distribution, therefore it is log-transformed to improve model performance.')
st. write('2. Data is read into a ML pipe using the sklearn framework. Text in video titles and tags is tokenized using a CountVectorizer, excluding common stop words such as "the". An random forest model is used to predict log(like/dislike ratio) from title & tags text. The model was optimized using hyperparameter tuning.')
st. write('3. Model outputs are stored and used for visualization on a Heroku app. Although the model could be used to predict like/dislike ratio from other videos, we will use it to assess which words in the video titles and tags were the most polarizing. To do this, we extract the feature importances from random forest. The features (i.e. words) with the most importance are those which the model selects as having the most influence on predicting the like/dislike ratio, therefore we can assess them as the most polarizing.')
st. write('4. The Heroku app allows interactive exploration of these results. The app displays model performance, and allows users to explore the most polarizing words from the dataset. Some interesting results: topical subjects such as "trump" and "news" apear in the top 5 most polarizing words, reflecting the engagement towards politics and news outlets of the later 2010s. "funny" is ranked n. 7, and videos mentioning this word tend to have above-average log(like/dislike) ratio (Fig. 2). Surprisingly, the term "bowl" is ranked 9th most polarizing - upon closer inspection of these videos (Table 2), most of them refer to the February 2018 Super Bowl.')


# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
