#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[2]:


FOTR_sentiments=pd.read_csv(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_sentiments.csv')
FOTR_sentiments=FOTR_sentiments.drop(columns=['Unnamed: 0'])
FOTR_sentiments.head()


# ## Create dataframe without prologue

# In[25]:


FOTR_sentiments_nopro = FOTR_sentiments.drop(FOTR_sentiments.index[[0,1,2,3,4]])
FOTR_sentiments_nopro


# In[26]:


FOTR_sentiments_nopro.describe()


# In[4]:


import matplotlib.patches as mpatches

kwargs = dict(alpha=0.6)

patch1 = mpatches.Patch(label='Normalized: 3440 tokens', **kwargs)
patch2 = mpatches.Patch(color='b', label='Full token count', **kwargs)
all_handles = (patch1, patch2)

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_alpha(0.7)
ax.barh(FOTR_sentiments['Chapters'], FOTR_sentiments['Lex_density_norm'],alpha=.5)
ax.barh(FOTR_sentiments['Chapters'], FOTR_sentiments['Lex_density'],color='b',alpha=.7)
ax.set_title("Lexical Density by Chapter in The Fellowship of the Ring",fontsize=33)
ax.set_xlabel("Lexical Density Score", fontsize=27)
ax.set_ylabel("Chapters", fontsize=27)
#ax.set_xticklabels([-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25],fontsize=20)
ax.set_yticklabels(FOTR_sentiments.Chapters, rotation=0, fontsize=22)
ax.legend(handles=all_handles,loc='lower right', fontsize=22)
ax.tick_params(axis='x', which='major', labelsize=18)
ax.invert_yaxis()
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_Lexical_Diversity.png',bbox_inches='tight')


# In[5]:


import matplotlib.patches as mpatches

kwargs = dict(alpha=0.5)

patch1 = mpatches.Patch(color='g', label='Positive Rating (>0.05)', **kwargs)
patch2 = mpatches.Patch(color='r', label='Negative Rating (<-0.05)', **kwargs)
patch3 = mpatches.Patch(color='orange', label='Neutral Rating (-0.05 - 0.05)', **kwargs)
all_handles = (patch1, patch2, patch3)

fig, ax = plt.subplots(figsize=(15, 18))
ax.set_alpha(0.5)
ax.barh(FOTR_sentiments['Chapters'], FOTR_sentiments['Compound'],
        color=FOTR_sentiments.Rating.map({'Positive': 'g', 'Negative': 'r', 'Neutral': 'orange'}),
        alpha=.5)
ax.set_title("Sentiment Compound Scores in The Fellowship of the Ring (VADER)",fontsize=33)
ax.set_xlabel("Compound Score (Range= -1.0 - 1.0)", fontsize=27)
ax.set_ylabel("Chapters", fontsize=27)
#ax.set_xticklabels([-0.15,-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25],fontsize=20)
ax.set_yticklabels(FOTR_sentiments.Chapters, rotation=0, fontsize=22)
ax.legend(handles=all_handles,loc='lower right', fontsize=20)
ax.tick_params(axis='x', which='major', labelsize=18)
ax.invert_yaxis()
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_VADER_compound.png',bbox_inches='tight')


# In[28]:


print ('--- Sentiment Scores for The Fellowhip of the Ring Averaged Across Chapters without Prologue ---')
print ('\n')
print ('-- TEXT BLOB --')
print ('Polarity: {0:.3f}'.format(FOTR_sentiments_nopro['Polarity'].mean()))
print ('Subjectivity: {0:.3f}'.format(FOTR_sentiments_nopro['Subjectivity'].mean()))
print ('\n')
print ('-- VADER --')
print ('Positive: {0:.3f}'.format(FOTR_sentiments_nopro['Positive'].mean()))
print ('Negative: {0:.3f}'.format(FOTR_sentiments_nopro['Negative'].mean()))
print ('Neutral: {0:.3f}'.format(FOTR_sentiments_nopro['Neutral'].mean()))
print ('Compound: {0:.3f}'.format(FOTR_sentiments_nopro['Compound'].mean()))
print ('\n')
print ('-- NRC --')
print ('Positive: {0:.3f}'.format(FOTR_sentiments_nopro['Positive_NRC'].mean()))
print ('Joy: {0:.3f}'.format(FOTR_sentiments_nopro['Joy'].mean()))
print ('Anticipation: {0:.3f}'.format(FOTR_sentiments_nopro['Anticipation'].mean()))
print ('Surprise: {0:.3f}'.format(FOTR_sentiments_nopro['Surprise'].mean()))
print ('Trust: {0:.3f}'.format(FOTR_sentiments_nopro['Trust'].mean()))
print ('Negative: {0:.3f}'.format(FOTR_sentiments_nopro['Negative_NRC'].mean()))
print ('Anger: {0:.3f}'.format(FOTR_sentiments_nopro['Anger'].mean()))
print ('Fear: {0:.3f}'.format(FOTR_sentiments_nopro['Fear'].mean()))
print ('Disgust: {0:.3f}'.format(FOTR_sentiments_nopro['Disgust'].mean()))
print ('Sadness: {0:.3f}'.format(FOTR_sentiments_nopro['Sadness'].mean()))


# In[12]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Compound',figsize=(25,15), color='blue',
                             alpha=.35, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Negative',figsize=(25,15), color='red',
                             alpha=.35, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Neutral',figsize=(25,15), color='purple',
                             alpha=.35, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Positive',figsize=(25,15), color='green',
                             alpha=.35, linewidth=7, ax=ax)
plt.axhline(y=0, xmin=0, xmax=1, alpha=.5, color='orange', linestyle='--', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Fellowship of the Ring (VADER)', fontsize=40)
plt.xlim(-1,27)
plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks([-0.2,0,0.2,0.4,0.6,0.8,1.0],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_VADER_full.png',bbox_inches='tight')


# In[29]:


ax1 = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Compound',figsize=(25,10), color='blue',
                             alpha=.35, linewidth=7, ax=ax1)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Negative',figsize=(25,10), color='red',
                             alpha=.35, linewidth=7, ax=ax1)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Positive',figsize=(25,10), color='green',
                             alpha=.35, linewidth=7, ax=ax1)
plt.axhline(y=0, xmin=0, xmax=1, alpha=.5, color='orange', linestyle='--', linewidth=5)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Fellowship of the Ring (VADER: without "Neutral")', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.5,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks([-0.15,-0.10,-0.05,0,0.05,0.10,0.15,0.20,0.25],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_VADER_zoom.png',bbox_inches='tight')


# In[14]:


ax2 = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Polarity',figsize=(25,10), color='blue',
                             alpha=.35,linewidth=7, ax=ax2)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Subjectivity',figsize=(25,10), color='orange', 
                             alpha=.35, linewidth=7, ax=ax2)
plt.legend(loc='best', fontsize=25)
plt.title('Polarity/Subjectivity of The Fellowship of the Ring (TextBlob)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.5,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5],fontsize=20)
plt.ylabel('Average Sentiment', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_TextBlob_polarity.png',bbox_inches='tight')


# In[15]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Positive_NRC',figsize=(25,15), color='green',
                             alpha=.35, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Negative_NRC',figsize=(25,15), color='red',
                             alpha=.35, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_PosNeg.png',bbox_inches='tight')


# In[16]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Anger',figsize=(25,15), color='red',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Disgust',figsize=(25,15), color='purple',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Fear',figsize=(25,15), color='maroon',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Sadness',figsize=(25,15), color='black',
                             alpha=.45, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment (Negative) of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_Neg.png',bbox_inches='tight')


# In[17]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Anticipation',figsize=(25,15), color='blue',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Joy',figsize=(25,15), color='orange',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Surprise',figsize=(25,15), color='lightblue',
                             alpha=.45, linewidth=7, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Trust',figsize=(25,15), color='green',
                             alpha=.45, linewidth=7, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Chapter Sentiment (Positive) of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_Pos.png',bbox_inches='tight')


# In[18]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Sadness',figsize=(25,15), color='blue',
                             alpha=.35, linewidth=8, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Joy',figsize=(25,15), color='yellow',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Joy/Sadness Sentiments of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_JoySad.png',bbox_inches='tight')


# In[19]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Anticipation',figsize=(25,15), color='orange',
                             alpha=.35, linewidth=8, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Surprise',figsize=(25,15), color='teal',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Surprise/Anticipation Sentiments of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_SurAnt.png',bbox_inches='tight')


# In[20]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Trust',figsize=(25,15), color='green',
                             alpha=.35, linewidth=8, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Disgust',figsize=(25,15), color='purple',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Trust/Disgust Sentiments of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_TrustDis.png',bbox_inches='tight')


# In[21]:


ax = plt.gca()
FOTR_sentiments.plot(kind='line',x='Chapters', y='Anger',figsize=(25,15), color='red',
                             alpha=.35, linewidth=8, ax=ax)
FOTR_sentiments.plot(kind='line',x='Chapters', y='Fear',figsize=(25,15), color='forestgreen',
                             alpha=.35, linewidth=8, ax=ax)
plt.legend(loc='best', fontsize=20)
plt.title('Anger/Fear Sentiments of The Fellowship of the Ring (NRC)', fontsize=40)
plt.xlim(-1,27)
#plt.ylim(-.2,1)
plt.xlabel('Chapter', fontsize=30)
plt.xticks(np.arange(27), FOTR_sentiments.Chapters[0:27], rotation=75, ha='right',fontsize=25)
plt.yticks(fontsize=16)
plt.ylabel('Sentiment Score', fontsize=30)
#plt.show()

plt.savefig(r'C:\Users\Nick\Desktop\GitProjects\NLP_projects\FOTR\FOTR_NRC_AngerFear.png',bbox_inches='tight')


# In[ ]:




