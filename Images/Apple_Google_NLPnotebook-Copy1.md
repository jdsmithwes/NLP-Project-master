
# Project 4: NLP - Sentiment Analysis
 Jamaal Smith<br>
 Flatiron School


## Introduction

Sentiment Analysis is a useful tool within the Natural Language Processing domain that enables organizations to get a sense how the public is responding to their brand. In this analysis, I will perform such an analysis on a set of tweets from [data.world.](https://data.world/crowdflower/brands-and-product-emotions).

Through analysis of these tweets, I hope to learn the context with which Apple and Google are being compared in these tweets. By this, are these tweets all from a single point in time such as a product release. I also believe analysis of the tweets can provide context as to the tweet's audience base. Finally, this analysis can also provide useful insights into how the brand with lesser social media on clout might be able to revamp their image to drive the sort of customer engagement that they desire.

# Obtaining Data


```python
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
sns.set_context('talk')
```


```python
data = pd.read_csv('data.csv',encoding='utf-8')
data = data.astype(str)
tweets = pd.DataFrame(data['tweet_text'])
```

## Scrubbing/Cleaning Data

### DataFrame treatment


```python
#preview of data
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <td>1</td>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>2</td>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>3</td>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <td>4</td>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>5</td>
      <td>@teachntech00 New iPad Apps For #SpeechTherapy...</td>
      <td>nan</td>
      <td>No emotion toward brand or product</td>
    </tr>
    <tr>
      <td>6</td>
      <td>nan</td>
      <td>nan</td>
      <td>No emotion toward brand or product</td>
    </tr>
    <tr>
      <td>7</td>
      <td>#SXSW is just starting, #CTIA is around the co...</td>
      <td>Android</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Beautifully smart and simple idea RT @madebyma...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Counting down the days to #sxsw plus strong Ca...</td>
      <td>Apple</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
</div>




```python
#stats on data
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>9288</td>
      <td>9288</td>
      <td>9288</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>9168</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <td>top</td>
      <td>nan</td>
      <td>nan</td>
      <td>No emotion toward brand or product</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>27</td>
      <td>5997</td>
      <td>5389</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get names of indexes for which column Age has value 30
indexNames = data[data['emotion_in_tweet_is_directed_at'] == 'nan' ].index
# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)

data.dropna()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>emotion_in_tweet_is_directed_at</th>
      <th>is_there_an_emotion_directed_at_a_brand_or_product</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>iPhone</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <td>1</td>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>2</td>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>3</td>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>iPad or iPhone App</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <td>4</td>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>9272</td>
      <td>@mention your PR guy just convinced me to swit...</td>
      <td>iPhone</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>9274</td>
      <td>&amp;quot;papyrus...sort of like the ipad&amp;quot; - ...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>9275</td>
      <td>Diller says Google TV &amp;quot;might be run over ...</td>
      <td>Other Google product or service</td>
      <td>Negative emotion</td>
    </tr>
    <tr>
      <td>9280</td>
      <td>I've always used Camera+ for my iPhone b/c it ...</td>
      <td>iPad or iPhone App</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>9283</td>
      <td>Ipad everywhere. #SXSW {link}</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
  </tbody>
</table>
<p>3291 rows × 3 columns</p>
</div>




```python
data=data.rename(columns={'emotion_in_tweet_is_directed_at':'Brand','is_there_an_emotion_directed_at_a_brand_or_product':'Sentiment'})
```


```python
#stats without nan in tweet column
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>Brand</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>3291</td>
      <td>3291</td>
      <td>3291</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>3281</td>
      <td>9</td>
      <td>4</td>
    </tr>
    <tr>
      <td>top</td>
      <td>RT @mention Marissa Mayer: Google Will Connect...</td>
      <td>iPad</td>
      <td>Positive emotion</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>3</td>
      <td>946</td>
      <td>2672</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Changing column info to assist with encoding later
product_dict = {"iPhone":'Apple','iPad or iPhone App': 'Apple','iPad':'Apple','nan':'none','Android':'Google','Android App':'Google','Other Google product or service':'Google','Other Apple product or service':'Apple'}
sentiment_dict = {'Negative emotion': '-1','Positive emotion':'1','No emotion toward brand or product':'0', "I can't tell": '0'}
data = data.replace({"Brand": product_dict})
data = data.replace({'Sentiment':sentiment_dict})

#data['prediction'] = (data['Brand'],data['Sentiment'])
data['prediction'] = list(zip(data.Brand, data.Sentiment))
print(len(data['Brand']))
print(len(data['Sentiment']))
```

    3291
    3291



```python
from sklearn.preprocessing import OneHotEncoder
brands = data.Brand.as_matrix(columns=None)
brands = brands.reshape(-1,1)
oe = OneHotEncoder()
data['Brand_code']= oe.fit_transform(brands)
```

    /opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>Brand</th>
      <th>Sentiment</th>
      <th>prediction</th>
      <th>Brand_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>.@wesley83 I have a 3G iPhone. After 3 hrs twe...</td>
      <td>Apple</td>
      <td>-1</td>
      <td>(Apple, -1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>@jessedee Know about @fludapp ? Awesome iPad/i...</td>
      <td>Apple</td>
      <td>1</td>
      <td>(Apple, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>@swonderlin Can not wait for #iPad 2 also. The...</td>
      <td>Apple</td>
      <td>1</td>
      <td>(Apple, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>@sxsw I hope this year's festival isn't as cra...</td>
      <td>Apple</td>
      <td>-1</td>
      <td>(Apple, -1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>@sxtxstate great stuff on Fri #SXSW: Marissa M...</td>
      <td>Google</td>
      <td>1</td>
      <td>(Google, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
    </tr>
  </tbody>
</table>
</div>



### Twitter data


```python
data.head(10) #preview of what data looks like
tweets = data['tweet_text']
len(tweets)
```




    3291




```python
import re
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
stopword_list = stopwords.words('english')
stopword_list.append([',','rt','mention','link'])
for tweet in tweets:
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.join([c for c in tweet if c not in punctuation])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in stopword_list]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/jamaalsmith/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
data['tweet_text'] = corpus #so that dataframe has cleaned tweets
```


```python
#Creation of Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

data['Sentiment'] = data['Sentiment'].to_numpy(dtype=int,copy=False)
y = (data['Sentiment'])
y1 = data['Brand']

```


```python
##Tokenize data and then generate FreqDist
from nltk import word_tokenize
tokens = word_tokenize(','.join(map(str,corpus)))
```


```python
#FreqDist
from nltk import FreqDist
freq = FreqDist(tokens)
most_used_words = freq.most_common(100)
most_used_words = pd.DataFrame(most_used_words, columns=['word','count'])
most_used_words.reset_index(drop=True,inplace=True)

```

## Exploratory Data Analysis


```python
#Wordcloud of popular words
text = " ".join(tweet for tweet in data.tweet_text)
plt.figure(figsize=(35,10))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["mention", "link", "rt",])

# Generate a word cloud image
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](images/output_24_0.png)


Based on the wordcloud,it appears that the tweets reviewed were taken at a SXSW event in Austin, TX. Further, there is specific mention of the words ipad, apple store and launch. When I Googled ipad product launches at SXSW, I learned that Apple launched its iPad2 product at the 2011 event. 

With the knowledge that these tweets were centered around the SXSW festival, I then noticed that what appeared to be two sets of names were in the top twenty list. One name appears to be Marissa Mayer and the other was Tim Reilli. I performed another Google and [found information about the following event in 2011.](https://www.mediabullseye.com/2011/03/sxsw-update-chasing-the-ideas-of-innovation/) This information further assists us with learning about our the author of the tweets we are reviewing because both presenters spoke to advances that their respective organizations had made with location based services at this forum.

I then examined the events that Google had planned for the 2011 SXSW festival. I undertook this query because I noticed that Google was in the top 20 most frequently used words. During the 2011 SXSW event, [Google had a plethora of events that touched on subjects such as recommendation engines and hybrid marketing.](https://googleblog.blogspot.com/2011/03/google-at-sxsw-2011-austin-here-we-come.html) Based on this quick research, one can assume that tweets related to Google will be commenting on the new technologies the firm was presenting at this conference.


Finally, it is worth noting some initial impressions about the tweet's larger context. After reading some of the tweets before preprocessing, I noticed that some of the tweets related to Apple appeared to be focused on the user experience people were having with apple products at SXSW. I then noticed that like Google, at this time, Apple was launching its iPad2. Thus, it is safe to assume that tweets would be a good reflection the sentiment that festival goers had related to these launches.

***

This analyis will not simply report back whether more individuals favored one company to another. Instead, its findings can provide insight into how users of their product's found their latest offerings when first presented with them at a technology conference. 



```python
#top 10 words
top10_words = most_used_words[:10]
top10_words.head()

plt.figure(figsize=(10,5))
sns.barplot(x='word',y='count',data=top10_words)
plt.title('Top 10 Words')
```




    Text(0.5, 1.0, 'Top 10 Words')




![png](images/output_26_1.png)


### Popularity of the Two Brands


```python
#Count of how many times each Brand is mentioned
plt.figure(figsize=(10,5))
sns.countplot(x="Brand",data=data)
plt.title('Popularity of Company based on Tweets')
plt.xlabel('Company')
```




    Text(0.5, 0, 'Company')




![png](images/output_28_1.png)


Despite only having two inputs (Apple & Other Apple product or service) when compared to Google that had four inputs (Google,Android, Android App, Other Google product or service), Apple was discussed more in the tweets according to the data.

One might conclude that the launch of the iPad2 was a major event that Apple marketed well and built suspense amongst the tech community.


```python
#Sentiment distribution for each Brand
plt.figure(figsize=(10,5))
sns.countplot(x='Brand',hue='Sentiment',data=data)
plt.title('Attendee Sentiment Expressed in Tweets')
```




    Text(0.5, 1.0, 'Attendee Sentiment Expressed in Tweets')




![png](images/output_30_1.png)



```python
#from textblob import textblob
#data['polarity'] = data['tweet_text'].map(lambda text: TextBlob(text).sentiment.polarity)
data['review_len'] = data['tweet_text'].astype(str).apply(len)
data['word_count'] = data['tweet_text'].apply(lambda x: len(str(x).split()))
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_text</th>
      <th>Brand</th>
      <th>Sentiment</th>
      <th>prediction</th>
      <th>Brand_code</th>
      <th>review_len</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>wesley g iphon hr tweet rise austin dead need ...</td>
      <td>Apple</td>
      <td>-1</td>
      <td>(Apple, -1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
      <td>9285</td>
      <td>1682</td>
    </tr>
    <tr>
      <td>1</td>
      <td>jessede know fludapp awesom ipad iphon app lik...</td>
      <td>Apple</td>
      <td>1</td>
      <td>(Apple, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
      <td>11967</td>
      <td>2070</td>
    </tr>
    <tr>
      <td>2</td>
      <td>swonderlin wait ipad also sale sxsw swonderlin...</td>
      <td>Apple</td>
      <td>1</td>
      <td>(Apple, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
      <td>2861</td>
      <td>495</td>
    </tr>
    <tr>
      <td>3</td>
      <td>sxsw hope year festiv crashi year iphon app sx...</td>
      <td>Apple</td>
      <td>-1</td>
      <td>(Apple, -1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
      <td>4017</td>
      <td>729</td>
    </tr>
    <tr>
      <td>4</td>
      <td>sxtxstate great stuff fri sxsw marissa mayer g...</td>
      <td>Google</td>
      <td>1</td>
      <td>(Google, 1)</td>
      <td>(0, 0)\t1.0\n  (1, 0)\t1.0\n  (2, 0)\t1.0\n ...</td>
      <td>14145</td>
      <td>2263</td>
    </tr>
  </tbody>
</table>
</div>




```python
#review length compared against tweets
plt.figure(figsize=(10,5))
sns.stripplot(x='Brand',y='word_count',hue='Sentiment',data=data,edgecolor='gray',jitter=True)
plt.xlabel('Brand')
plt.ylabel('Word Count')
plt.title('Length of Tweets by Brand')
```




    Text(0.5, 1.0, 'Length of Tweets by Brand')




![png](images/output_33_1.png)



```python
#Average length of tweets by Brand
brand_length = data.groupby(['Brand','Sentiment']).mean()
brand_length.round(decimals=2)

data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentiment</th>
      <th>review_len</th>
      <th>word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>3291.000000</td>
      <td>3291.000000</td>
      <td>3291.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.654208</td>
      <td>7952.615618</td>
      <td>1337.426922</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.736064</td>
      <td>3754.438144</td>
      <td>632.980208</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-1.000000</td>
      <td>47.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1.000000</td>
      <td>5000.500000</td>
      <td>831.500000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1.000000</td>
      <td>7956.000000</td>
      <td>1333.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.000000</td>
      <td>10729.000000</td>
      <td>1792.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>19975.000000</td>
      <td>3361.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
brand_length=pd.DataFrame(brand_length)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>review_len</th>
      <th>word_count</th>
    </tr>
    <tr>
      <th>Brand</th>
      <th>Sentiment</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" valign="top">Apple</td>
      <td>-1</td>
      <td>7947.592784</td>
      <td>1326.672680</td>
    </tr>
    <tr>
      <td>0</td>
      <td>8094.222222</td>
      <td>1378.375000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7616.916367</td>
      <td>1299.615187</td>
    </tr>
    <tr>
      <td rowspan="3" valign="top">Google</td>
      <td>-1</td>
      <td>9032.931298</td>
      <td>1472.816794</td>
    </tr>
    <tr>
      <td>0</td>
      <td>7220.964286</td>
      <td>1185.607143</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8678.751037</td>
      <td>1422.398340</td>
    </tr>
  </tbody>
</table>
</div>



## Modeling

### Final Pre-Processing


```python
#Encoding
from sklearn.preprocessing import OneHotEncoder

oe = OneHotEncoder()
oe.fit(X)
```

    /opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/preprocessing/_encoders.py:415: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)





    OneHotEncoder(categorical_features=None, categories=None, drop=None,
                  dtype=<class 'numpy.float64'>, handle_unknown='error',
                  n_values=None, sparse=True)



### Naive Bayes Model


```python
#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```


```python
#training Naive Bayes for sentiment
from sklearn.naive_bayes import GaussianNB
classifier_sentiment = GaussianNB()
classifier_sentiment.fit(X_train, y_train)


```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
#splitting the data
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.20, random_state = 0)
```


```python
#training Naive Bayes for Brand
from sklearn.naive_bayes import GaussianNB
classifier_brand = GaussianNB()
classifier_brand.fit(X1_train, y1_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
#predictions
y_pred_sentiment = classifier_sentiment.predict(X_test)
y_pred_brand = classifier_brand.predict(X_test)
```


```python
#evaluation of sentiment model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_sentiment)
print(cm)
accuracy_score(y_test, y_pred_sentiment)
```

    [[ 60   3  39]
     [  5   1  11]
     [160  32 348]]





    0.6206373292867982




```python
#confusion matrix visual for sentiment
cm = confusion_matrix(y_test,y_pred_sentiment)
acc = accuracy_score(y_test,y_pred_sentiment)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.6206373292867982





    Text(0.5, 1, 'Confusion Matrix')




![png](images/output_46_2.png)



```python
#confusion matrix visual for Brand
cm = confusion_matrix(y1_test,y_pred_brand)
acc = accuracy_score(y1_test,y_pred_brand)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.7207890743550834





    Text(0.5, 1, 'Confusion Matrix')




![png](images/output_47_2.png)


### Recurrent Neural Network - LSTM


```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding

model = Sequential()

#Embedding layer
model.add(
    Embedding(input_dim=1200,output_dim=10,
              trainable=True,
              mask_zero=True))

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(1, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1,recurrent_activation='sigmoid'))

# Fully connected layer
model.add(Dense(300, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.1))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Create callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=5,min_delta=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)]
```


```python
#Training Model
history = model.fit(X_train,  y_train,
                     epochs=2, callbacks = callbacks,validation_data=(X_train, y_train))
```

    Epoch 1/2
    83/83 [==============================] - 197s 2s/step - loss: 7.7676e-08 - accuracy: 0.5448 - val_loss: 7.7676e-08 - val_accuracy: 0.7884
    Epoch 2/2
    83/83 [==============================] - 188s 2s/step - loss: 7.7676e-08 - accuracy: 0.5418 - val_loss: 7.7676e-08 - val_accuracy: 0.7884



```python
#Predictions
y_pred = model.predict(X_test)
```


```python
#Evaluation
LTSM_eval = model.evaluate(X_test,y_test)
```

    21/21 [==============================] - 3s 143ms/step - loss: 7.9232e-08 - accuracy: 0.7967



```python
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```

    103/103 [==============================] - 12s 119ms/step - loss: 7.7988e-08 - accuracy: 0.7900
    
    accuracy: 79.00%


## Insights/Findings

- The popularity of the ipad2 launch at SXSW definitely contributed to more tweets during this period related to Apple. Despite not having the same type of product launch, events held by Google and presentations by individuals such as Marissa Mayer helped Google remain relevant.
- Individuals that tweeted about Google had lengthier tweets than their Apple counterparts. This is especially for tweets with negative sentiments related to Google. On the flip side, when individuals wrote lengthy tweets related to Apple, they were most likely to share a positive sentiment.
- Our machine learning model was able to produce an accuracy rate at predicting a tweet's of only 62%. Utilizing the Recurrent Neural Network model allowed us to predict a tweet's sentiment with an accuracy of 80%
- While the ipad2 launch was the big mover for Apple, Google also garnered traction with what appears to be an app that has something to the do with location-based services. This information can be gleaned from the word map. The word map also can inform us that lots of tweets were commenting on events in Downtown Austin, making mention of pop-up and other forms of temporary shops. As far as sentiments, the word maps that tweeters often used when discussing affinity to items were: cool, great, update, and major. With this in mind, these were probably what Apple strove to create in the public during its public build-up of the iPad2 launch.
- Ultimately the success each company achieved at the 2011 SXSW festival can only be determined when one knows what their marketing goal was. For Apple, if their goal was to create buzz and positive initial reviews for the iPad2, it appears that they accomplished this goal. For Google, if they wanted to impress avid tech goers with their latest developments with apps and other research, they may have also been successful based on the wordcloud.



```python
data.head()
data.groupby('Brand').sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sentiment</th>
      <th>review_len</th>
      <th>word_count</th>
    </tr>
    <tr>
      <th>Brand</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apple</td>
      <td>1561</td>
      <td>18511820</td>
      <td>3146942</td>
    </tr>
    <tr>
      <td>Google</td>
      <td>592</td>
      <td>7660238</td>
      <td>1254530</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
