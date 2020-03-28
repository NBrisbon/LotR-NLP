#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.text import Text
from nltk.corpus import brown
import requests
from bs4 import BeautifulSoup
nltk.download('popular')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[381]:


r = requests.get('https://archive.org/stream/TheLordOfTheRing1TheFellowshipOfTheRing/The+Lord+Of+The+Ring+1-The+Fellowship+Of+The+Ring_djvu.txt')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print(html[:2000])


# In[382]:


# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

# Fixing mistakes in text
text = text.replace('N®menor', 'Numenor')
text = text.replace('H®rin', 'Hurin')
text = text.replace('T®rin', 'Turin')
text = text.replace('D®nedain', 'Dunedain')
text = text.replace('ores', 'orcs')
text = text.replace('Khazad-dym', 'Khazad-dum')
text = text.replace('And®ril', 'Anduril') 
text = text.replace('Lothlurien', 'Lothlorien')
text = text.replace('Lurien', 'Lorien')
text = text.replace('Forcst', 'Forest')


# Entire Book
FOTR = text[16363:1021945]

  ## PROLONGUE ##

# Concerning Hobbits
p1 = FOTR[0:19430] 

# Concerning Pipe-weed
p2 = FOTR[19436:22791]

# Of the Ordering of the Shire
p3 = FOTR[22797:27493]

# Of the Finding of the Ring
p4 = FOTR[27499:36075]

# NOTE ON THE SHIRE RECORDS
p5 = FOTR[36081:41360]

  ## BOOK 1 ##

# A Long Expected Party
b1_1 = FOTR[41381:96384]

# The Shadow of the Past
b1_2 = FOTR[96390:156092]

# Three is Company
b1_3 = FOTR[156098:207998]

# A Short Cut to Mushrooms
b1_4 = FOTR[208004:239136]

# A Conspiracy Unmasked
b1_5 = FOTR[239142:266920]

# The Old Forest
b1_6 = FOTR[266925:302280]

# In the House of Tom Bombadil
b1_7 = FOTR[302286:332328]

# Fog on the Barrow-Downs 
b1_8 = FOTR[332334:368502]

# At the Sign of The Prancing Pony 
b1_9 = FOTR[368508:402448]

# Strider
b1_10 = FOTR[402453:433350]

# A Knife in the Dark
b1_11 = FOTR[433356:484433]

# Flight to the Ford
b1_12 = FOTR[484438:531603]

  ## BOOK 2 ##
    
# Many Meetings
b2_1 = FOTR[531624:580340]

# The Council of Elrond
b2_2 = FOTR[580347:666210]

# The Ring Goes South
b2_3 = FOTR[666215:723395]

# A Journey in the Dark
b2_4 = FOTR[723400:785328]

# The Bridge of Khazad-dum
b2_5 = FOTR[785334:814630]

# Lothlorien
b2_6 = FOTR[814636:865064]

# The Mirror of Galadriel
b2_7 = FOTR[865069:901276]

# Farewell to Lorien
b2_8 = FOTR[901282:934245]

# The Great River
b2_9 = FOTR[934251:972952]

#The Breaking of the Fellowship 
b2_10 = FOTR[972958:]


#print(b1_6)


# In[337]:


# Find specific spots in the text
FOTR.find("The Breaking of the Fellowship")


# In[383]:


texts = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

data = pd.DataFrame(data=[text for text in texts], columns=['Text'])

display(data.head(27))


# In[384]:


chapters = ['Concerning Hobbits', 'Concerning Pipe-weed', 'Of the Ordering of the Shire', 'Of the Finding of the Ring',
           'NOTE ON THE SHIRE RECORDS', 'A Long-expected Party', 'The Shadow of the Past', 'Three is Company',
           'A Short Cut to Mushrooms', 'A Conspiracy Unmasked', 'The Old Forest', 'In the House of Tom Bombadil',
           'Fog on the Barrow-Downs', 'At the Sign of The Prancing Pony', 'Strider', 'A Knife in the Dark', 
           'Flight to the Ford', 'Many Meetings', 'The Council of Elrond', 'The Ring Goes South', 'A Journey in the Dark',
           'The Bridge of Khazad-dum', 'Lothlorien', 'The Mirror of Galadriel', 'Farewell to Lorien', 'The Great River',
           'The Breaking of the Fellowship']

book = ['Prologue', 'Prologue', 'Prologue', 'Prologue','Prologue', 'Book I', 'Book I', 'Book I','Book I', 
        'Book I', 'Book I', 'Book I','Book I', 'Book I', 'Book I', 'Book I', 'Book I', 'Book II', 'Book II', 
        'Book II', 'Book II','Book II', 'Book II', 'Book II', 'Book II', 'Book II','Book II']

data1 = pd.DataFrame(chapters, columns=['Chapters'])

data1_1 = pd.merge(data, data1, left_index=True, right_index=True)

data1_2 = pd.DataFrame(book, columns=['Book'])

data2 = pd.merge(data1_1, data1_2, left_index=True, right_index=True)


display(data2.head(27))


# In[385]:


data2['Volume']='FOTR'
data2 = data2[['Volume', 'Book', 'Chapters', 'Text']]
data2


# In[386]:


import pandas as pd
from nltk import tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

chapters = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

analyzer = SentimentIntensityAnalyzer()

sentiments_list = list()

for chapter in chapters:
    sentence_list = tokenize.sent_tokenize(chapter)
    sentiments = {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        sentiments['compound'] += vs['compound']
        sentiments['neg'] += vs['neg']
        sentiments['neu'] += vs['neu']
        sentiments['pos'] += vs['pos']
            
    sentiments['compound'] = sentiments['compound'] / len(sentence_list)
    sentiments['neg'] = sentiments['neg'] / len(sentence_list)
    sentiments['neu'] = sentiments['neu'] / len(sentence_list)
    sentiments['pos'] = sentiments['pos'] / len(sentence_list)
    
    sentiments_list.append(sentiments)  # add this line

data3 = pd.DataFrame(sentiments_list)  # add this line
data3


# In[387]:


FOTR_sentiments = pd.merge(data2, data3, left_index=True, right_index=True)

FOTR_sentiments=FOTR_sentiments.rename(columns={"compound": "Compound", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments


# In[388]:


from textblob import TextBlob

blob_book = TextBlob(FOTR)
blob_p1 = TextBlob(p1)
blob_p2 = TextBlob(p2)
blob_p3 = TextBlob(p3)
blob_p4 = TextBlob(p4)
blob_p5 = TextBlob(p5)
blob_b1_1 = TextBlob(b1_1)
blob_b1_2 = TextBlob(b1_2)
blob_b1_3 = TextBlob(b1_3)
blob_b1_4 = TextBlob(b1_4)
blob_b1_5 = TextBlob(b1_5)
blob_b1_6 = TextBlob(b1_6)
blob_b1_7 = TextBlob(b1_7)
blob_b1_8 = TextBlob(b1_8)
blob_b1_9 = TextBlob(b1_9)
blob_b1_10 = TextBlob(b1_10)
blob_b1_11 = TextBlob(b1_11)
blob_b1_12 = TextBlob(b1_12)
blob_b2_1 = TextBlob(b2_1)
blob_b2_2 = TextBlob(b2_2)
blob_b2_3 = TextBlob(b2_3)
blob_b2_4 = TextBlob(b2_4)
blob_b2_5 = TextBlob(b2_5)
blob_b2_6 = TextBlob(b2_6)
blob_b2_7 = TextBlob(b2_7)
blob_b2_8 = TextBlob(b2_8)
blob_b2_9 = TextBlob(b2_9)
blob_b2_10 = TextBlob(b2_10)


# In[389]:


blobs = [blob_p1, blob_p2, blob_p3, blob_p4, blob_p5, blob_b1_1, blob_b1_2, blob_b1_3, blob_b1_4, blob_b1_5,
        blob_b1_6, blob_b1_7, blob_b1_8, blob_b1_9, blob_b1_10, blob_b1_11, blob_b1_12, blob_b2_1, blob_b2_2,
        blob_b2_3, blob_b2_4, blob_b2_5, blob_b2_6, blob_b2_7, blob_b2_8, blob_b2_9, blob_b2_10]

polarity_list = list()
subjectivity_list = list()

for blob in blobs:
    sentence_list1 = blob.sentences
    polarity = (0)
    subjectivity = (0)
        
    for sentence in sentence_list1:
        pl=sentence.sentiment.polarity
        polarity += pl
        sb=sentence.sentiment.subjectivity
        subjectivity += sb
            
    polarity = polarity / len(sentence_list1)
    subjectivity = subjectivity / len(sentence_list1)

    polarity_list.append(polarity)  
    subjectivity_list.append(subjectivity)

data4 = pd.DataFrame(polarity_list)  
data5 = pd.DataFrame(subjectivity_list)  
data6 = pd.merge(data4, data5, left_index=True, right_index=True)
data6=data6.rename(columns={'0_x': "Polarity", '0_y': 'Subjectivity'})

data6


# In[390]:


FOTR_sentiments = pd.merge(FOTR_sentiments, data6, left_index=True, right_index=True)

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments


# In[391]:


chapters = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

wordlength_list = list()

for chapter in chapters:
    count = len(chapter)
        
    wordlength_list.append(count)  

data7 = pd.DataFrame(wordlength_list)  
data7=data7.rename(columns={0: 'Total_words'})
data7


# In[392]:


FOTR_sentiments = pd.merge(FOTR_sentiments, data7, left_index=True, right_index=True)

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments.head()


# In[393]:


def rating(FOTR_sentiments):
    if FOTR_sentiments['Compound'] > 0.05:
        return 'Positive'
    elif FOTR_sentiments['Compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

FOTR_sentiments['Rating'] = FOTR_sentiments.apply(rating, axis=1)

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments.head(27)


# In[369]:


import io, re, nltk
from pathlib import Path
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import FreqDist

chapters = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

Noun_freq = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    stop = set(stopwords.words('english'))
    words = [word for word in tokens if word not in stop]
    tagged = pos_tag(words)
    nouns = [word for word, pos in tagged if (pos == 'NN')]
    Noun1 = (0)
            
    Noun1 = FreqDist(nouns).most_common(10)
    
    Noun_freq.append(Noun1)
    
Noun_freq


# In[394]:


# The Lexical Diversity represents the ratio of unique words used to the total number of words in the story.

chapters = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

def lexical_density(text):
    return len(set(text)) / len(text)

Lexical_density = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    LexDen = (0)
            
    LexDen = lexical_density(tokens)

    Lexical_density.append(LexDen)  

LexDen = pd.DataFrame(Lexical_density) 
LexDen=LexDen.rename(columns={0: 'Lex_density'})
LexDen


# In[395]:


# The Lexical Diversity represents the ratio of unique words used to the total number of words in the story.

chapters = [p1,p2,p3,p4,p5,b1_1,b1_2,b1_3,b1_4,b1_5,b1_6,b1_7,b1_8,b1_9,b1_10,b1_11,b1_12,b2_1,b2_2,b2_3,
         b2_4,b2_5,b2_6,b2_7,b2_8,b2_9,b2_10]

def lexical_density(text):
    return len(set(text)) / len(text)

Lexical_density_norm = list()

for chapter in chapters:
    chap = chapter
    data = chap.replace('\n', ' ')
    data_lower = data.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(data_lower)
    tokens = tokens[0:27778]
    LexDen1 = (0)
            
    LexDen1 = lexical_density(tokens)

    Lexical_density_norm.append(LexDen1)  

LexDen_norm = pd.DataFrame(Lexical_density_norm) 
LexDen_norm=LexDen_norm.rename(columns={0: 'Lex_density_norm'})
LexDen_norm


# In[396]:


FOTR_sentiments = pd.merge(FOTR_sentiments, LexDen, left_index=True, right_index=True)

FOTR_sentiments = FOTR_sentiments[['Volume','Book','Chapters','Text','Total_words','Lex_density','Negative',
                                                  'Neutral','Positive','Compound','Rating','Polarity','Subjectivity']]

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments.head()


# In[397]:


FOTR_sentiments = pd.merge(FOTR_sentiments, LexDen_norm, left_index=True, right_index=True)

FOTR_sentiments = FOTR_sentiments[['Volume','Book','Chapters','Text','Total_words','Lex_density','Lex_density_norm','Negative',
                                                  'Neutral','Positive','Compound','Rating','Polarity','Subjectivity']]

FOTR_sentiments.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments.head()


# In[399]:


FOTR_sentiments = FOTR_sentiments.replace('\n',' ', regex=True) 
FOTR_sentiments.head()


# In[400]:


import pandas as pd
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm_notebook as tqdm
from tqdm import trange


def text_emotion(df, column):
    '''
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns for each emotion
    '''

    new_df = df.copy()

    xlsx = pd.read_excel(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\The_Silmarillion\NRC-Sentiment-Emotion-Lexicons\NRC-Sentiment-Emotion-Lexicons\NRC-Emotion-Lexicon-v0.92\NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx')
    emolex_df = xlsx[['word', 'Positive_NRC','Negative_NRC','Anger', 'Anticipation', 'Disgust', 'Fear','Joy',
                      'Sadness', 'Surprise', 'Trust']]
    emotions = emolex_df.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")

    
    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        for i, row in new_df.iterrows():
            pbar.update(1)
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_df[emolex_df.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df


# In[401]:


FOTR_sentiments_final = text_emotion(FOTR_sentiments, 'Text')


# In[402]:


FOTR_sentiments_final.head()


# In[403]:


FOTR_sentiments_final['word_count'] = FOTR_sentiments_final['Text'].apply(tokenize.word_tokenize).apply(len)
FOTR_sentiments_final.head()


# In[404]:


FOTR_sentiments_final.dtypes


# In[405]:


emotions = ['Positive_NRC','Negative_NRC','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise',
                 'Trust']


# In[406]:


for emotion in emotions:
    FOTR_sentiments_final[emotion]=FOTR_sentiments_final[emotion]/FOTR_sentiments_final['word_count']

FOTR_sentiments_final.head()


# In[407]:


def ratings(FOTR_sentiments_final):
    if FOTR_sentiments_final['Compound'] > 0.05:
        return 1
    elif FOTR_sentiments_final['Compound'] < -0.05:
        return -1
    else:
        return 0

FOTR_sentiments_final['Rating_num'] = FOTR_sentiments_final.apply(ratings, axis=1)

FOTR_sentiments_final.to_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')

FOTR_sentiments_final.head(10)


# In[ ]:




