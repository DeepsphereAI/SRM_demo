#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.toktok import ToktokTokenizer
from text_preprocessing import clean_review
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from collections import Counter
import re


# In[2]:

model = joblib.load(open('model_nb.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))

# In[3]:

sentiment_map = {'Negative':0, 'Positive':1}

# In[6]:

tokenizer = ToktokTokenizer()
sia = SentimentIntensityAnalyzer()

# In[7]:

def get_sentiment(text):
    """
    Predicts the sentiment of text using the Multinomial Naive Bayes Model
    """
    sentiment_id = model.predict(cv.transform([text]).toarray())
    return get_name(sentiment_id)

# In[8]:

def get_name(sentiment_id):
    """
    Gets sentiment name from sentiment_map using sentiment_id
    """
    for sentiment, id_ in sentiment_map.items():
        if id_ == sentiment_id:
            return sentiment

# In[9]:

def get_noun(text):
    """
    Finds noun of the text
    """
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)    
    pos_tags = nltk.pos_tag(tokens)
    nouns = []
    for word, tag in pos_tags:
        if tag == "NN" or tag == "NNP" or tag == "NNS":
            nouns.append(word)
    return nouns


# In[10]:


def get_tokens(text):
    """
    Converts text to a list of tokens using nltk tokenizer
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


# In[11]:


def top_pos_word(text):
    """
    Finds top positive word using nltk vader library
    """
    pos_polarity = dict()
    for word in get_tokens(text):
        pos_score = sia.polarity_scores(word)['pos']
        if word not in pos_polarity:
            pos_polarity[word] = pos_score
        else:
            pos_polarity[word] += pos_score
    top_word = max(pos_polarity, key=pos_polarity.get)
    return top_word


# In[12]:


def top_neg_word(text):
    """
    Finds top negative word using nltk vader library
    """
    neg_polarity = dict()
    for word in get_tokens(text):
        neg_score = sia.polarity_scores(word)['neg']
        if word not in neg_polarity:
            neg_polarity[word] = neg_score
        else:
            neg_polarity[word] += neg_score
    top_word = max(neg_polarity, key=neg_polarity.get)
    return top_word

def mmr(doc_embedding, word_embeddings, words, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def frequency_counter(sentence):
    """
    Finds frequency of words in cleaned text using BERT keyword extraction 
    """
    
    sentence =" ".join(sentence)
    cleaned_text = clean_review(sentence)
    print(cleaned_text)
    
    n_gram_range = (3, 3)
    stop_words = "english"

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([cleaned_text])
    candidates = count.get_feature_names()

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([cleaned_text])
    candidate_embeddings = model.encode(candidates)
    
    sentence = mmr(doc_embedding, candidate_embeddings, candidates, top_n=20, diversity=0.1)
    
    sentence_list= (' '.join(sentence)).split()
    keywords = ((" ".join(sorted(set(sentence_list), key=sentence_list.index)))).split()
    words = cleaned_text.split()
    
    to_search = re.compile('|'.join(sorted(keywords, key=len, reverse=True)))
    matches = (to_search.search(el) for el in words)
    counts = Counter(match.group() for match in matches if match)
    count_word = dict(counts)
  
    word_freq_df = (pd.DataFrame([count_word]).T).reset_index()
    
    word_freq_df.columns = word_freq_df.columns.map(str)
    word_freq_df.rename(columns = {'index':'word', '0':'count'}, inplace = True)
    word_freq_df = word_freq_df.sort_values(by=['count'], ascending=False)
   
    return word_freq_df
# In[15]:

def sentiment_analysis(texts):
    """
    Finds the sentiment of text, sentiment score of cleaned text 
    and stores the result to dataframe
    
    """
    sentiment_list = []
    vader_score = []
    for text in texts:
        text = clean_review(text)
        #print(f'Cleaning Reviews ............')
        sentiment = get_sentiment(text)
        #print(f'Classified Sentiment for the review :{sentiment}')
        sentiment_list.append(sentiment)
        sid_obj = SentimentIntensityAnalyzer() 
        sentiment_dict = sid_obj.polarity_scores(text) 
        vader_score.append(sentiment_dict["compound"])
        #print(vader_score)
        
        #print(f'Sentiment: {sentiment}')
        
    pred_dict = {'Reviews':texts, 'Sentiment':sentiment_list,'Sentiment_Score':vader_score}
    pred_data = pd.DataFrame(pred_dict,columns=['Reviews','Sentiment','Sentiment_Score'])
    pred_data['Ranking'] = pred_data['Sentiment_Score'].rank(ascending = 1) # rank based on sentiment score
    pred_data['Ranking'] = pred_data['Ranking'].apply(int)
    pred_data.sort_values("Ranking", inplace = True)
    pred_data.drop('Sentiment_Score',axis=1,inplace=True)
    return pred_data






