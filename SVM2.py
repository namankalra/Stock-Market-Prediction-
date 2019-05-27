import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
import matplotlib.pyplot as plt

#New Dataset
news = pd.read_csv('News.csv')
news['Top1'][0]=re.sub(r'\'', ' ' ,news['Top1'][0])
news['Top1'][0]

#pre processing
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    text = re.sub(r'b ', '', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# Clean the headlines
from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
nltk.download('vader_lexicon')

sia = SIA()
results=[]
rslt = pd.DataFrame(index=range(0,len(news)),columns=['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'])

for i in range(len(news)):
    news['Top1'][i]=clean_text(news['Top1'][i])
    pol_score = sia.polarity_scores(news['Top1'][i])
    results.append(pol_score)
rslt['1']=results;
results=[]
for i in range(len(news)):
    news['Top2'][i]=clean_text(news['Top2'][i])
    pol_score = sia.polarity_scores(news['Top2'][i])
    results.append(pol_score)
rslt['2']=results;
results=[]
for i in range(len(news)):
    news['Top3'][i]=clean_text(news['Top3'][i])
    pol_score = sia.polarity_scores(news['Top3'][i])
    results.append(pol_score)
rslt['3']=results;
results=[]
for i in range(len(news)):
    news['Top4'][i]=clean_text(news['Top4'][i])
    pol_score = sia.polarity_scores(news['Top4'][i])
    results.append(pol_score)
rslt['4']=results;
results=[]
for i in range(len(news)):
    news['Top5'][i]=clean_text(news['Top5'][i])
    pol_score = sia.polarity_scores(news['Top5'][i])
    results.append(pol_score)
rslt['5']=results;
results=[]
for i in range(len(news)):
    news['Top6'][i]=clean_text(news['Top6'][i])
    pol_score = sia.polarity_scores(news['Top6'][i])
    results.append(pol_score)
rslt['6']=results;
results=[]
for i in range(len(news)):
    news['Top7'][i]=clean_text(news['Top7'][i])
    pol_score = sia.polarity_scores(news['Top7'][i])
    results.append(pol_score)
rslt['7']=results;
results=[]
for i in range(len(news)):
    news['Top8'][i]=clean_text(news['Top8'][i])
    pol_score = sia.polarity_scores(news['Top8'][i])
    results.append(pol_score)
rslt['8']=results;
results=[]
for i in range(len(news)):
    news['Top9'][i]=clean_text(news['Top9'][i])
    pol_score = sia.polarity_scores(news['Top9'][i])
    results.append(pol_score)
rslt['9']=results;
results=[]
for i in range(len(news)):
    news['Top10'][i]=clean_text(news['Top10'][i])
    pol_score = sia.polarity_scores(news['Top10'][i])
    results.append(pol_score)
rslt['10']=results;
results=[]
for i in range(len(news)):
    news['Top11'][i]=clean_text(news['Top11'][i])
    pol_score = sia.polarity_scores(news['Top11'][i])
    results.append(pol_score)
rslt['11']=results;
results=[]
for i in range(len(news)):
    news['Top12'][i]=clean_text(news['Top12'][i])
    pol_score = sia.polarity_scores(news['Top12'][i])
    results.append(pol_score)
rslt['12']=results;
results=[]
for i in range(len(news)):
    news['Top13'][i]=clean_text(news['Top13'][i])
    pol_score = sia.polarity_scores(news['Top13'][i])
    results.append(pol_score)
rslt['13']=results;
results=[]
for i in range(len(news)):
    news['Top14'][i]=clean_text(news['Top14'][i])
    pol_score = sia.polarity_scores(news['Top14'][i])
    results.append(pol_score)
rslt['14']=results;
results=[]
for i in range(len(news)):
    news['Top15'][i]=clean_text(news['Top15'][i])
    pol_score = sia.polarity_scores(news['Top15'][i])
    results.append(pol_score)
rslt['15']=results;
results=[]
for i in range(len(news)):
    news['Top16'][i]=clean_text(news['Top16'][i])
    pol_score = sia.polarity_scores(news['Top16'][i])
    results.append(pol_score)
rslt['16']=results;
results=[]
for i in range(len(news)):
    news['Top17'][i]=clean_text(news['Top17'][i])
    pol_score = sia.polarity_scores(news['Top17'][i])
    results.append(pol_score)
rslt['17']=results;
results=[]
for i in range(len(news)):
    news['Top18'][i]=clean_text(news['Top18'][i])
    pol_score = sia.polarity_scores(news['Top18'][i])
    results.append(pol_score)
rslt['18']=results;
results=[]
for i in range(len(news)):
    news['Top19'][i]=clean_text(news['Top19'][i])
    pol_score = sia.polarity_scores(news['Top19'][i])
    results.append(pol_score)
rslt['19']=results;
results=[]
for i in range(len(news)):
    news['Top20'][i]=clean_text(news['Top20'][i])
    pol_score = sia.polarity_scores(news['Top20'][i])
    results.append(pol_score)
rslt['20']=results;
results=[]
for i in range(len(news)):
    news['Top21'][i]=clean_text(news['Top21'][i])
    pol_score = sia.polarity_scores(news['Top21'][i])
    results.append(pol_score)
rslt['21']=results;
results=[]
for i in range(len(news)):
    news['Top22'][i]=clean_text(news['Top22'][i])
    pol_score = sia.polarity_scores(news['Top22'][i])
    results.append(pol_score)
rslt['22']=results;
results=[]
for i in range(len(news)):
    news['Top23'][i]=clean_text(news['Top23'][i])
    pol_score = sia.polarity_scores(news['Top23'][i])
    results.append(pol_score)
rslt['23']=results;
results=[]
for i in range(len(news)):
    news['Top24'][i]=clean_text(news['Top24'][i])
    pol_score = sia.polarity_scores(news['Top24'][i])
    results.append(pol_score)
rslt['24']=results;
results=[]
for i in range(len(news)):
    news['Top25'][i]=clean_text(news['Top25'][i])
    pol_score = sia.polarity_scores(news['Top25'][i])
    results.append(pol_score)
rslt['25']=results;
tya=rslt['1'][0]['compound']+rslt['3'][0]['compound']
tya
results=[]
X=pd.DataFrame(index=range(0,len(news)),columns=['Data1','Data2'])


for i in range(len(rslt)):
    tya=rslt['1'][i]['compound']+rslt['2'][i]['compound']+rslt['3'][i]['compound']+rslt['4'][i]['compound']+rslt['5'][i]['compound']+rslt['6'][i]['compound']+rslt['7'][i]['compound']+rslt['8'][i]['compound']+rslt['9'][i]['compound']+rslt['10'][i]['compound']+rslt['12'][i]['compound']+rslt['13'][i]['compound']+rslt['14'][i]['compound']+rslt['15'][i]['compound']+rslt['16'][i]['compound']+rslt['17'][i]['compound']+rslt['18'][i]['compound']+rslt['19'][i]['compound']+rslt['20'][i]['compound']+rslt['21'][i]['compound']+rslt['22'][i]['compound']
    X['Data1'][i]=tya

# iterating over rows using iterrows() function  

# 
#for i in range(0,len(news)):
 #   news[Top1][i]=re.strip('b""')

for i in range(len(news)):
    X['Data2'][i]=news['Label'][i]
    
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X['Data1'],X['Data2'],random_state=42,
                                               test_size=0.2,shuffle=False)

x_train=X_train.values
y_train=Y_train.values
x_test=X_test.values
y_test=Y_test.values

x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

from sklearn.svm import SVC

#Gaussian kernel
svr_rbf = SVC(kernel='rbf')

svr_rbf.fit(x_train,y_train)

import matplotlib.pyplot as plt  
plt.scatter(x_test,y_test, label= "stars", color= "green",  
            marker= "*", s=30) 
plt.scatter(x_test,y_rbf, label= "stars", color= "red",  
            marker= "*", s=30) 

plt.xlabel('Days')
plt.ylabel('Value') 

plt.title('Stock Market Prediciton')

plt.legend() 

plt.show() 

#SvM

from sklearn.svm import SVR

#Gaussian kernel
svr_rbf = SVR(kernel='rbf')

svr_rbf.fit(x_train,y_train)


Y_predsvm=svr_rbf.predict(x_test)
Y_pred=[]
for i in range(len(Y_predsvm)):
    if Y_predsvm[i]>=0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)

import numpy
Ypred=numpy.asarray(Y_pred)
from sklearn.metrics import confusion_matrix
cmsvm=confusion_matrix(y_test,Ypred)

from sklearn import metrics
scoresvm=metrics.accuracy_score(Y_test,Y_pred)
print("accuracy:   %0.3f" % scoresvm)
