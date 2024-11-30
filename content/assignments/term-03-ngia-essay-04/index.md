---
author: ["Shweata N. Hegde"]
title: "Term 03 | The New Geography of Information Age | Extension Notes 04: Anti-vaxxers---Where Are They'" 
date: "2024-11-17"
description: ""
tags: ["assignments", "term-03"]
ShowToc: true
---

# **Anti-Vaxxers -- Where Are They?**
Decades of research show that vaccines save lives. However, there are communities of people known as anti-vaxxers who hesitate or refuse to get vaccinated. When faced with devastating pandemics like COVID-19, this "vaccine hesitancy" poses a threat to humanity. The advent of social media has only complicated the issue, providing opportunities for anti-vaxxers to spread misinformation.

As part of the Extension Notes, I looked at YouTube comments on:
1. A [short video](https://www.youtube.com/shorts/joUqPs2imcw) about the rare side effects of COVID-19 vaccines by a news channel.
2. An [explainer video](https://www.youtube.com/watch?v=zBkVCpbNnkU) about the side effects of vaccines by a science communication channel.

The goal was to explore the ratio of pro-vaccinators to anti-vaxxers in the comments for these videos. A natural assumption would be that the news video on rare side effects would attract more anti-vaxxers than the explainer. This also means that viewers of explainers are more likely to be the ones who already trust vaccines, whereas the majority of anti-vaxxers would not even click on it.

To test my assumption, I used sentiment analysis, "the process of analyzing large volumes of text to determine whether it expresses a positive sentiment, a negative sentiment, or a neutral sentiment" [IBM]. The YouTube API was used to retrieve the top 100 comments for each of the two videos.

## Sentiment Analysis -- VanderSentiment and BERT


```python
!pip install -q google-api-python-client
!pip install -q transformers
!pip install -q vaderSentiment
!pip install -q emoji
```


```python
from google.colab import userdata
import os
import googleapiclient.discovery
from transformers import pipeline
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import emoji
```


```python
# Disable OAuthlib's HTTPS verification when running locally.
# *DO NOT* leave this option enabled in production.
def get_comments(videoID):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = userdata.get('youtube-ngia')

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=videoID,
        maxResults = 100
    )
    response = request.execute()

    comment_list = []
    for item in response['items']:
        comment_list.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    return comment_list
```


```python
rare_side_effect_comments = get_comments("joUqPs2imcw")
print(len(rare_side_effect_comments))
```

    100



```python
side_effect_explainer_comments = get_comments("zBkVCpbNnkU")
print(len(side_effect_explainer_comments))
```

    100



```python
def cleanup(comments_list):
    hyperlink_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    threshold_ratio = 0.65

    relevant_comments = []

    # Inside your loop that processes comments
    for comment_text in comments_list:

        comment_text = comment_text.lower().strip()

        emojis = emoji.emoji_count(comment_text)

        # Count text characters (excluding spaces)
        text_characters = len(re.sub(r'\s', '', comment_text))

        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)
    return relevant_comments
```


```python
cleaned_comment_list_rare_side_effects = cleanup(rare_side_effect_comments)
print(len(cleaned_comment_list_rare_side_effects))
```

    100



```python
cleaned_comment_list_side_effect_explainer = cleanup(side_effect_explainer_comments)
print(len(cleaned_comment_list_side_effect_explainer))
```

    90



```python
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_rse_news = pipe(cleaned_comment_list_rare_side_effects)
sentiment_rse_news
```

    Some weights of the model checkpoint at cardiffnlp/twitter-roberta-base-sentiment-latest were not used when initializing RobertaForSequenceClassification: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).





    [{'label': 'negative', 'score': 0.6061795353889465},
     {'label': 'negative', 'score': 0.5674803256988525},
     {'label': 'negative', 'score': 0.5921235084533691},
     {'label': 'neutral', 'score': 0.4312528371810913},
     {'label': 'positive', 'score': 0.5960631966590881},
     {'label': 'negative', 'score': 0.6562612056732178},
     {'label': 'negative', 'score': 0.9345315098762512},
     {'label': 'neutral', 'score': 0.5495634078979492},
     {'label': 'negative', 'score': 0.8604071140289307},
     {'label': 'neutral', 'score': 0.7570379972457886},
     {'label': 'negative', 'score': 0.7719420790672302},
     {'label': 'negative', 'score': 0.79497230052948},
     {'label': 'negative', 'score': 0.7235072255134583},
     {'label': 'positive', 'score': 0.8092665076255798},
     {'label': 'negative', 'score': 0.9118055105209351},
     {'label': 'positive', 'score': 0.5191245675086975},
     {'label': 'positive', 'score': 0.6608207821846008},
     {'label': 'negative', 'score': 0.8262534737586975},
     {'label': 'neutral', 'score': 0.5972841382026672},
     {'label': 'negative', 'score': 0.8186414241790771},
     {'label': 'neutral', 'score': 0.6172197461128235},
     {'label': 'negative', 'score': 0.8782438635826111},
     {'label': 'neutral', 'score': 0.645826518535614},
     {'label': 'negative', 'score': 0.8831338882446289},
     {'label': 'negative', 'score': 0.7845684885978699},
     {'label': 'neutral', 'score': 0.8189308047294617},
     {'label': 'neutral', 'score': 0.6449963450431824},
     {'label': 'positive', 'score': 0.5297830700874329},
     {'label': 'negative', 'score': 0.909889817237854},
     {'label': 'negative', 'score': 0.48418086767196655},
     {'label': 'negative', 'score': 0.9658225178718567},
     {'label': 'negative', 'score': 0.6825441122055054},
     {'label': 'neutral', 'score': 0.5192729830741882},
     {'label': 'neutral', 'score': 0.6761555671691895},
     {'label': 'positive', 'score': 0.7686967849731445},
     {'label': 'negative', 'score': 0.6796764731407166},
     {'label': 'negative', 'score': 0.8358374834060669},
     {'label': 'negative', 'score': 0.521063506603241},
     {'label': 'negative', 'score': 0.7262055277824402},
     {'label': 'neutral', 'score': 0.8232724666595459},
     {'label': 'positive', 'score': 0.8965514898300171},
     {'label': 'negative', 'score': 0.8366003036499023},
     {'label': 'neutral', 'score': 0.8434333205223083},
     {'label': 'negative', 'score': 0.7170586585998535},
     {'label': 'negative', 'score': 0.7483528256416321},
     {'label': 'negative', 'score': 0.8972554206848145},
     {'label': 'neutral', 'score': 0.7339447736740112},
     {'label': 'neutral', 'score': 0.5544359683990479},
     {'label': 'neutral', 'score': 0.7739880084991455},
     {'label': 'negative', 'score': 0.9182223081588745},
     {'label': 'negative', 'score': 0.49541255831718445},
     {'label': 'negative', 'score': 0.7888889312744141},
     {'label': 'negative', 'score': 0.7534485459327698},
     {'label': 'negative', 'score': 0.8910378217697144},
     {'label': 'neutral', 'score': 0.6409167647361755},
     {'label': 'negative', 'score': 0.711844265460968},
     {'label': 'negative', 'score': 0.5471321940422058},
     {'label': 'neutral', 'score': 0.5378387570381165},
     {'label': 'negative', 'score': 0.851806104183197},
     {'label': 'neutral', 'score': 0.9008152484893799},
     {'label': 'neutral', 'score': 0.7890174388885498},
     {'label': 'negative', 'score': 0.7292648553848267},
     {'label': 'negative', 'score': 0.8771555423736572},
     {'label': 'neutral', 'score': 0.5564010143280029},
     {'label': 'negative', 'score': 0.5493753552436829},
     {'label': 'negative', 'score': 0.5761932134628296},
     {'label': 'neutral', 'score': 0.8509829044342041},
     {'label': 'negative', 'score': 0.8043684363365173},
     {'label': 'neutral', 'score': 0.7026739120483398},
     {'label': 'negative', 'score': 0.5961050391197205},
     {'label': 'negative', 'score': 0.8648480176925659},
     {'label': 'neutral', 'score': 0.6092495322227478},
     {'label': 'positive', 'score': 0.9110750555992126},
     {'label': 'positive', 'score': 0.4348452389240265},
     {'label': 'negative', 'score': 0.5091502070426941},
     {'label': 'neutral', 'score': 0.5459365248680115},
     {'label': 'negative', 'score': 0.6213790774345398},
     {'label': 'negative', 'score': 0.5243015289306641},
     {'label': 'neutral', 'score': 0.541743278503418},
     {'label': 'negative', 'score': 0.8069548606872559},
     {'label': 'negative', 'score': 0.6447361707687378},
     {'label': 'neutral', 'score': 0.4813714921474457},
     {'label': 'neutral', 'score': 0.6734002828598022},
     {'label': 'neutral', 'score': 0.6803257465362549},
     {'label': 'negative', 'score': 0.8159052133560181},
     {'label': 'neutral', 'score': 0.6945080161094666},
     {'label': 'neutral', 'score': 0.542612612247467},
     {'label': 'neutral', 'score': 0.6104950308799744},
     {'label': 'neutral', 'score': 0.8620831370353699},
     {'label': 'neutral', 'score': 0.6719599366188049},
     {'label': 'negative', 'score': 0.6729378700256348},
     {'label': 'neutral', 'score': 0.5041062235832214},
     {'label': 'negative', 'score': 0.8897811770439148},
     {'label': 'negative', 'score': 0.832440197467804},
     {'label': 'negative', 'score': 0.7506840825080872},
     {'label': 'neutral', 'score': 0.6282171607017517},
     {'label': 'negative', 'score': 0.6919422149658203},
     {'label': 'positive', 'score': 0.810357928276062},
     {'label': 'negative', 'score': 0.5888343453407288},
     {'label': 'negative', 'score': 0.7692237496376038}]




```python
# prompt: count the number of positive, negative and neutral labels of sentiment bad

positive_count = 0
negative_count = 0
neutral_count = 0

for sentiment in sentiment_rse_news:
  label = sentiment['label']
  if label == 'positive':
    positive_count += 1
  elif label == 'negative':
    negative_count += 1
  elif label == 'neutral':
    neutral_count += 1

print(f"Positive Labels: {positive_count}")
print(f"Negative Labels: {negative_count}")
print(f"Neutral Labels: {neutral_count}")
```

    Positive Labels: 10
    Negative Labels: 55
    Neutral Labels: 35



```python
for i, element in enumerate(cleaned_comment_list_side_effect_explainer):
   if len(element) > 500:
     cleaned_comment_list_side_effect_explainer[i] = element[:500]
sentiment_rse_explainer = pipe(cleaned_comment_list_side_effect_explainer)
sentiment_rse_explainer
```




    [{'label': 'positive', 'score': 0.8019840717315674},
     {'label': 'negative', 'score': 0.8265782594680786},
     {'label': 'neutral', 'score': 0.9340046644210815},
     {'label': 'neutral', 'score': 0.632490336894989},
     {'label': 'negative', 'score': 0.8450339436531067},
     {'label': 'negative', 'score': 0.8585911989212036},
     {'label': 'negative', 'score': 0.8621253371238708},
     {'label': 'neutral', 'score': 0.7768394947052002},
     {'label': 'neutral', 'score': 0.6206957101821899},
     {'label': 'negative', 'score': 0.7979456782341003},
     {'label': 'negative', 'score': 0.6610816717147827},
     {'label': 'negative', 'score': 0.8289511799812317},
     {'label': 'negative', 'score': 0.7712651491165161},
     {'label': 'neutral', 'score': 0.625948429107666},
     {'label': 'neutral', 'score': 0.5452887415885925},
     {'label': 'neutral', 'score': 0.8207365274429321},
     {'label': 'positive', 'score': 0.9801045656204224},
     {'label': 'negative', 'score': 0.4834241569042206},
     {'label': 'neutral', 'score': 0.4456772208213806},
     {'label': 'neutral', 'score': 0.6914815306663513},
     {'label': 'neutral', 'score': 0.9375398755073547},
     {'label': 'negative', 'score': 0.6118802428245544},
     {'label': 'neutral', 'score': 0.6936677694320679},
     {'label': 'negative', 'score': 0.7109909057617188},
     {'label': 'neutral', 'score': 0.7639383673667908},
     {'label': 'positive', 'score': 0.9580971598625183},
     {'label': 'negative', 'score': 0.8257519602775574},
     {'label': 'positive', 'score': 0.8790163993835449},
     {'label': 'neutral', 'score': 0.7752033472061157},
     {'label': 'negative', 'score': 0.7372997403144836},
     {'label': 'neutral', 'score': 0.6656520962715149},
     {'label': 'neutral', 'score': 0.6393928527832031},
     {'label': 'negative', 'score': 0.6549505591392517},
     {'label': 'neutral', 'score': 0.7818443179130554},
     {'label': 'negative', 'score': 0.8262605667114258},
     {'label': 'positive', 'score': 0.9833112955093384},
     {'label': 'neutral', 'score': 0.7258610725402832},
     {'label': 'negative', 'score': 0.8705864548683167},
     {'label': 'positive', 'score': 0.9609665870666504},
     {'label': 'positive', 'score': 0.5984934568405151},
     {'label': 'neutral', 'score': 0.49663740396499634},
     {'label': 'positive', 'score': 0.9839099049568176},
     {'label': 'positive', 'score': 0.9025787115097046},
     {'label': 'negative', 'score': 0.9034289717674255},
     {'label': 'neutral', 'score': 0.5653076171875},
     {'label': 'neutral', 'score': 0.7628281712532043},
     {'label': 'neutral', 'score': 0.858733594417572},
     {'label': 'neutral', 'score': 0.651912271976471},
     {'label': 'negative', 'score': 0.7870814204216003},
     {'label': 'negative', 'score': 0.4895855486392975},
     {'label': 'neutral', 'score': 0.7993780970573425},
     {'label': 'neutral', 'score': 0.5675607323646545},
     {'label': 'neutral', 'score': 0.6230518817901611},
     {'label': 'negative', 'score': 0.7396385073661804},
     {'label': 'negative', 'score': 0.8413233757019043},
     {'label': 'positive', 'score': 0.8745176196098328},
     {'label': 'neutral', 'score': 0.6338366270065308},
     {'label': 'negative', 'score': 0.7463333010673523},
     {'label': 'negative', 'score': 0.8111202716827393},
     {'label': 'positive', 'score': 0.7582007646560669},
     {'label': 'neutral', 'score': 0.6068226099014282},
     {'label': 'positive', 'score': 0.9560517072677612},
     {'label': 'negative', 'score': 0.8112869262695312},
     {'label': 'negative', 'score': 0.9552446603775024},
     {'label': 'negative', 'score': 0.9110667705535889},
     {'label': 'negative', 'score': 0.9388836622238159},
     {'label': 'positive', 'score': 0.9572903513908386},
     {'label': 'negative', 'score': 0.5817897319793701},
     {'label': 'neutral', 'score': 0.5996477603912354},
     {'label': 'neutral', 'score': 0.8453704714775085},
     {'label': 'neutral', 'score': 0.7800437808036804},
     {'label': 'neutral', 'score': 0.8104947209358215},
     {'label': 'positive', 'score': 0.820480227470398},
     {'label': 'negative', 'score': 0.49667373299598694},
     {'label': 'negative', 'score': 0.6832389235496521},
     {'label': 'negative', 'score': 0.6727829575538635},
     {'label': 'neutral', 'score': 0.5654632449150085},
     {'label': 'negative', 'score': 0.7574722766876221},
     {'label': 'negative', 'score': 0.739123523235321},
     {'label': 'neutral', 'score': 0.620697021484375},
     {'label': 'neutral', 'score': 0.692396879196167},
     {'label': 'negative', 'score': 0.5664587616920471},
     {'label': 'positive', 'score': 0.813576340675354},
     {'label': 'positive', 'score': 0.8674988746643066},
     {'label': 'neutral', 'score': 0.6593059301376343},
     {'label': 'positive', 'score': 0.7860581278800964},
     {'label': 'negative', 'score': 0.5511854290962219},
     {'label': 'negative', 'score': 0.7801781892776489},
     {'label': 'neutral', 'score': 0.700764536857605},
     {'label': 'negative', 'score': 0.830989420413971}]




```python
positive_count = 0
negative_count = 0
neutral_count = 0

for sentiment in sentiment_rse_explainer:
  label = sentiment['label']
  if label == 'positive':
    positive_count += 1
  elif label == 'negative':
    negative_count += 1
  elif label == 'neutral':
    neutral_count += 1

print(f"Positive Labels: {positive_count}")
print(f"Negative Labels: {negative_count}")
print(f"Neutral Labels: {neutral_count}")
```

    Positive Labels: 17
    Negative Labels: 37
    Neutral Labels: 36



```python
def make_csv(sentiment, comment_list, name_of_file):
    df = pd.DataFrame(sentiment, comment_list)
    print(df)
    df.to_csv(name_of_file)
```


```python
make_csv(sentiment_rse_news, cleaned_comment_list_rare_side_effects, "news_comments.csv")
```

                                                           label     score
    the coronavirus disease 2019, covid-19 (abbrevi...  negative  0.606180
    should&#39;ve said &quot;few knew&quot;.            negative  0.567480
    so glad i didn&#39;t get the vaccine.               negative  0.592124
    glad me and my family had enough sense to stay ...   neutral  0.431253
    remember everyone &quot;safe and effective&quot;    positive  0.596063
    ...                                                      ...       ...
    largest vaccine safety study ever - your first ...   neutral  0.628217
    it&#39;s rare because it&#39;s rare. a few mill...  negative  0.691942
    always cool and happy to blood donation daily i...  positive  0.810358
    when you’re 30 years old, exercise regularly an...  negative  0.588834
    its interesting after the first two shots my he...  negative  0.769224
    
    [100 rows x 2 columns]



```python
make_csv(sentiment_rse_explainer, cleaned_comment_list_side_effect_explainer, "explainer_comments.csv")
```

                                                           label     score
    thank god i’m not a kid i’m 13 i’m practically ...  positive  0.801984
    looking at your videos i took vaccine thinking ...  negative  0.826578
    what about 2024?                                     neutral  0.934005
    alright, safe zone take 3. since no one replied...   neutral  0.632490
    a growing group of pathetic dumb people is what...  negative  0.845034
    ...                                                      ...       ...
    hehe                                                positive  0.786058
    i always have one question from all these vacci...  negative  0.551185
    not gonna lie kurzgesagt, partnering with the b...  negative  0.780178
    who is watching this in 2024                         neutral  0.700765
    dude cells have ptsd                                negative  0.830989
    
    [90 rows x 2 columns]



```python
def sentiment_scores(comment, polarity):

    # Creating a SentimentIntensityAnalyzer object.
    sentiment_object = SentimentIntensityAnalyzer()

    sentiment_dict = sentiment_object.polarity_scores(comment)
    polarity.append(sentiment_dict['compound'])

    return polarity


polarity = []
positive_comments = []
negative_comments = []
neutral_comments = []

for index, items in enumerate(cleaned_comment_list_rare_side_effects):
    polarity = sentiment_scores(items, polarity)

    if polarity[-1] > 0.05:
        positive_comments.append(items)
    elif polarity[-1] < -0.05:
        negative_comments.append(items)
    else:
        neutral_comments.append(items)

# Print polarity
print(len(positive_comments))
print(len(negative_comments))
print(len(neutral_comments))
```

    38
    34
    28



```python
f = open("ytcomments_bad.txt", 'w', encoding='utf-8')
for idx, comment in enumerate(cleaned_comment_list_side_effect_explainer):
    f.write(str(comment)+"\n")
f.close()
print("Comments stored successfully!")
```

    Comments stored successfully!


# NLTK


```python
import nltk
nltk.download('state_union')
nltk.download('punkt_tab')
```

    [nltk_data] Downloading package state_union to /root/nltk_data...
    [nltk_data]   Package state_union is already up-to-date!
    [nltk_data] Downloading package punkt_tab to /root/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!





    True




```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
```


```python
list_of_words = []
for item in cleaned_comment_list_rare_side_effects:
  tokens = [w for w in nltk.word_tokenize(item) if w.isalpha()]
  list_of_words.extend(tokens)
print(list_of_words)
#words = [w for w in cleaned_comment_list_bad if w.isalpha()]
#print(words)
finder = nltk.collocations.BigramCollocationFinder.from_words(list_of_words)
print(finder.ngram_fd.most_common(20))
```

    ['the', 'coronavirus', 'disease', 'abbreviation', 'of', 'the', 'english', 'words', 'coronavirus', 'disease', 'is', 'an', 'infectious', 'disease', 'caused', 'by', 'the', 'coronavirus', 'lit', 'the', 'infection', 'which', 'mainly', 'affects', 'the', 'respiratory', 'tract', 'was', 'first', 'described', 'in', 'at', 'the', 'end', 'of', 'wuhan', 'china', 'in', 'january', 'an', 'epidemic', 'of', 'this', 'disease', 'arose', 'in', 'china', 'and', 'later', 'the', 'disease', 'spread', 'to', 'other', 'countries', 'of', 'the', 'world', 'and', 'became', 'a', 'br', 'br', 'the', 'infection', 'is', 'usually', 'spread', 'by', 'airborne', 'droplets', 'in', 'certain', 'circumstances', 'the', 'disease', 'can', 'also', 'be', 'transmitted', 'through', 'aerosols', 'the', 'course', 'of', 'the', 'disease', 'is', 'manifests', 'in', 'various', 'ways', 'and', 'varies', 'greatly', 'sufferers', 'may', 'experience', 'symptoms', 'such', 'as', 'fever', 'dry', 'cough', 'fatigue', 'shortness', 'of', 'breath', 'sore', 'throat', 'runny', 'nose', 'sneezing', 'are', 'less', 'common', 'in', 'addition', 'to', 'asymptomatic', 'cases', 'of', 'the', 'disease', 'most', 'cases', 'show', 'mild', 'to', 'moderate', 'symptoms', 'but', 'sometimes', 'the', 'disease', 'can', 'progress', 'to', 'bilateral', 'pneumonia', 'or', 'failure', 'and', 'cause', 'death', 'in', 'addition', 'to', 'lung', 'damage', 'the', 'disease', 'has', 'been', 'observed', 'to', 'affect', 'the', 'liver', 'central', 'nervous', 'system', 'kidneys', 'blood', 'vessels', 'and', 'heart', 'recovery', 'depends', 'on', 'the', 'state', 'of', 'a', 'person', 's', 'immune', 'system', 'mortality', 'rates', 'are', 'estimated', 'to', 'be', 'between', 'and', 'but', 'vary', 'by', 'age', 'and', 'health', 'status', 'should', 've', 'said', 'quot', 'few', 'knew', 'quot', 'so', 'glad', 'i', 'didn', 't', 'get', 'the', 'vaccine', 'glad', 'me', 'and', 'my', 'family', 'had', 'enough', 'sense', 'to', 'stay', 'away', 'from', 'the', 'vaccine', 'remember', 'everyone', 'quot', 'safe', 'and', 'effective', 'quot', 'rare', 'listen', 'to', 'yourselves', 'no', 'they', 'effing', 'don', 't', 'covid', 'br', 'contract', 'open', 'door', 'void', 'from', 'but', 'youtube', 'silenced', 'anybody', 'who', 'ask', 'questions', 'youtube', 'should', 'be', 'held', 'liable', 'for', 'pushing', 'mis', 'information', 'immigrants', 'were', 'brought', 'here', 'to', 'administer', 'shots', 'n', 'results', 'to', 'replace', 'already', 'here', 'awake', 'yet', 'hid', 'side', 'effects', 'for', 'years', 'your', 'govt', 'imdemnified', 'big', 'pharma', 'vote', 'trump', 'rfk', 'will', 'head', 'the', 'dept', 'of', 'public', 'health', 'and', 'this', 'will', 'never', 'happen', 'in', 'the', 'usa', 'again', 'rare', 'risk', 'of', 'complications', 'sounds', 'good', 'until', 'your', 'the', 'one', 'suffering', 'from', 'the', 'side', 'effects', 'cure', 'the', 'complications', 'before', 'mandating', 'taking', 'the', 'poison', 'glad', 'i', 'didnt', 'take', 'the', 'risk', 'never', 'contracted', 'the', 'virus', 'just', 'practiced', 'caution', 'and', 'thorough', 'hygiene', 'i', 'm', 'still', 'so', 'glad', 'to', 'this', 'day', 'that', 'i', 'never', 'took', 'it', 'got', 'the', 'head', 'swelling', 'was', 'very', 'painful', 'could', 'feel', 'my', 'skull', 'moving', 'blood', 'in', 'wars', 'and', 'eyes', 'almost', 'popped', 'out', 'was', 'arching', 'in', 'my', 'bed', 'for', 'a', 'couple', 'days', 'i', 'have', 'shots', 'and', 'im', 'healthier', 'then', 'ppl', 'that', 'got', 'it', 'safe', 'and', 'effective', 'no', 'the', 'benefits', 'don', 't', 'outweigh', 'the', 'risks', 'when', 'alternatives', 'exist', 'also', 'the', 'serious', 'side', 'effects', 'seem', 'not', 'so', 'rare', 'at', 'all', 'quot', 'rare', 'side', 'effects', 'quot', 'yet', 'it', 'remains', 'br', 'quot', 'safe', 'and', 'effective', 'quot', 'br', 'the', 'only', 'thing', 'i', 'can', 'say', 'about', 'this', 'is', 'the', 'individuals', 'who', 'experienced', 'this', 'agreed', 'to', 'take', 'the', 'quot', 'magic', 'potion', 'quot', 'the', 'consequence', 'of', 'one', 's', 'actions', 'this', 'whole', 'study', 'violated', 'the', 'nuremberg', 'code', 'which', 'was', 'the', 'same', 'thing', 'we', 'charged', 'the', 'nazis', 'during', 'the', 'nuremberg', 'trials', 'there', 'finally', 'admitting', 'that', 'it', 'was', 'a', 'study', 'a', 'difference', 'from', 'the', 'term', 'safe', 'and', 'effective', 'because', 'if', 'it', 'was', 'safe', 'there', 's', 'no', 'reason', 'to', 'do', 'a', 'study', 'on', 'it', 'i', 'work', 'in', 'mental', 'health', 'and', 'have', 'had', 'two', 'people', 'in', 'the', 'past', 'two', 'months', 'talk', 'about', 'being', 'severely', 'ill', 'after', 'taking', 'the', 'covid', 'vaccine', 'one', 'of', 'them', 'had', 'a', 'brain', 'infection', 'i', 'had', 'one', 'jab', 'because', 'i', 'felt', 'pressured', 'then', 'i', 'ended', 'up', 'with', 'heart', 'issues', 'for', 'a', 'while', 'and', 'have', 'had', 'extremely', 'irregular', 'periods', 'every', 'since', 'it', 'was', 'so', 'bad', 'at', 'first', 'i', 'thought', 'i', 'was', 'seriously', 'unwell', 'i', 'put', 'two', 'and', 'two', 'together', 'realised', 'it', 'was', 'the', 'vaccine', 'and', 'never', 'had', 'another', 'the', 'word', 'rare', 'sure', 'is', 'used', 'a', 'lot', 'when', 'discussing', 'worked', 'in', 'healthcare', 'for', 'years', 'refused', 'to', 'get', 'my', 'family', 'vaccinated', 'with', 'this', 'for', 'a', 'multitude', 'of', 'reasons', 'number', 'one', 'being', 'my', 'kids', 'aren', 't', 'mommy', 'governments', 'lab', 'rats', 'also', 'big', 'red', 'flag', 'when', 'a', 'vaccine', 'is', 'forced', 'and', 'misinformation', 'is', 'the', 'only', 'thing', 'available', 'while', 'real', 'concerns', 'where', 'being', 'forcibly', 'denied', 'and', 'withheld', 'no', 'thank', 'you', 'man', 'if', 'we', 'think', 'about', 'these', 'statements', 'br', 'quot', 'the', 'benefits', 'vastly', 'outweigh', 'the', 'risks', 'who', 'in', 'their', 'right', 'mind', 'would', 'take', 'such', 'a', 'risk', 'inflammation', 'and', 'swelling', 'in', 'the', 'brain', 'and', 'spinal', 'cord', 'we', 'admit', 'folks', 'to', 'the', 'hospital', 'who', 'come', 'in', 'with', 'these', 'issues', 'the', 'medical', 'term', 'is', 'encephalitis', 'it', 'is', 'not', 'worth', 'it', 'they', 'knew', 'and', 'i', 'am', 'sure', 'others', 'knew', 'also', 'rare', 'is', 'becoming', 'not', 'so', 'rare', 'invest', 'in', 'corporate', 'held', 'funeral', 'stocks', 'business', 'is', 'going', 'to', 'pick', 'up', 'in', 'about', 'yrs', 'my', 'worse', 'worst', 'fear', 'liars', 'i', 'm', 'so', 'so', 'so', 'so', 'sad', 'that', 'i', 'got', 'the', 'shot', 'i', 'feel', 'so', 'stupid', 'i', 'was', 'brainwashed', 'there', 'are', 'no', 'benefits', 'to', 'taking', 'the', 'vaccines', 'if', 'you', 're', 'healthy', 'oh', 'yes', 'sure', 'ilk', 'right', 'out', 'and', 'get', 'a', 'jab', 'lol', 'not', 'and', 'yet', 'people', 'are', 'still', 'getting', 'boosters', 'i', 'was', 'recently', 'at', 'rite', 'aid', 'pharmacy', 'and', 'people', 'were', 'lining', 'up', 'to', 'get', 'it', 'nir', 'radiation', 'fixing', 'nutritional', 'deficiencies', 'd', 'zn', 'se', 'etc', 'exeercise', 'sufficient', 'sleep', 'reducing', 'comorbidities', 'etc', 'are', 'cheap', 'effective', 'otc', 'solutions', 'give', 'your', 'body', 'what', 'it', 'needs', 'amp', 'it', 'can', 'do', 'wonders', 'perhaps', 'you', 'could', 'refresh', 'my', 'memory', 'on', 'exactly', 'what', 'any', 'benefit', 'there', 'is', 'at', 'all', 'is', 'there', 'a', 'treatment', 'to', 'reverse', 'this', 'quot', 'brain', 'swelling', 'quot', 'and', 'is', 'it', 'covered', 'in', 'all', 'provinces', 'tired', 'of', 'being', 'told', 'about', 'quot', 'rare', 'quot', 'side', 'effects', 'that', 'i', 'shouldnt', 'be', 'worried', 'about', 'try', 'telling', 'that', 'to', 'the', 'people', 'suffering', 'with', 'them', 'if', 'they', 'really', 'cared', 'then', 'they', 'would', 'be', 'working', 'toward', 'finding', 'a', 'solution', 'to', 'reverse', 'these', 'types', 'of', 'injuries', 'both', 'my', 'sibling', 'and', 'i', 'have', 'pityriasis', 'lichenoides', 'mine', 'started', 'right', 'after', 'the', 'original', 'gardisil', 'theirs', 'from', 'the', 'moderna', 'i', 'am', 'still', 'being', 'told', 'to', 'this', 'day', 'that', 'it', 's', 'quot', 'just', 'eczema', 'quot', 'yet', 'my', 'sibling', 'had', 'the', 'severe', 'acute', 'type', 'of', 'pleva', 'and', 'ended', 'up', 'in', 'the', 'hospital', 'with', 'myocarditis', 'and', 'their', 'skin', 'rotting', 'off', 'and', 'almost', 'died', 'no', 'one', 'could', 'tell', 'us', 'why', 'it', 'happened', 'or', 'were', 'remotely', 'interested', 'in', 'finding', 'out', 'the', 'cause', 'and', 'diagnosing', 'them', 'aside', 'from', 'telling', 'us', 'it', 's', 'quot', 'eczema', 'quot', 'that', 'got', 'infected', 'not', 'a', 'single', 'biopsy', 'taken', 'from', 'either', 'of', 'us', 'to', 'confirm', 'their', 'diagnosis', 'the', 'treatment', 'for', 'pl', 'there', 'isnt', 'one', 'apparently', 'it', 'quot', 'goes', 'away', 'on', 'it', 's', 'own', 'quot', 'here', 'i', 'am', 'still', 'suffering', 'years', 'later', 'can', 't', 'even', 'get', 'covered', 'for', 'dupixent', 'in', 'bc', 'which', 'is', 'the', 'only', 'treatment', 'we', 'havent', 'tried', 'and', 'has', 'been', 'successful', 'in', 'treating', 'quot', 'severe', 'eczema', 'quot', 'though', 'i', 'doubt', 'it', 'will', 'do', 'anything', 'for', 'us', 'because', 'i', 'don', 't', 'have', 'quot', 'eczema', 'quot', 'it', 's', 'pl', 'if', 'you', 'or', 'anyone', 'you', 'know', 'has', 'sudden', 'onset', 'of', 'severe', 'dermatitis', 'that', 'started', 'after', 'their', 'vaccination', 'please', 'look', 'into', 'pityriasis', 'lichenoides', 'or', 'pleva', 'it', 'is', 'not', 'just', 'atopic', 'dermatitis', 'or', 'prurigo', 'nodularis', 'lots', 'of', 'moonshine', 'had', 'bad', 'effects', 'but', 'the', 'benefit', 'vastly', 'outweighed', 'the', 'risks', 'ridiculous', 'blood', 'donation', 'brings', 'oxygen', 'of', 'infinity', 'organs', 'in', 'body', 'of', 'humans', 'animals', 'is', 'called', 'blood', 'donation', 'information', 'quot', 'you', 'have', 'a', 'rare', 'complication', 'enjoy', 'your', 'benefits', 'quot', 'still', 'pushing', 'question', 'to', 'everyone', 'what', 'is', 'a', 'cold', 'did', 'the', 'flu', 'season', 'disappear', 'wake', 'up', 'america', 'big', 'pharma', 'and', 'the', 'government', 'knew', 'the', 'consequences', 'of', 'the', 'vaccine', 'trump', 'tray', 'to', 'worn', 'us', 'me', 'and', 'any', 'others', 'got', 'a', 'particular', 'cancer', 'affecting', 'lymph', 'nodes', 'after', 'i', 'think', 'we', 'should', 'unite', 'and', 'sue', 'ever', 'since', 'my', 'shot', 'which', 'i', 'didn', 't', 'even', 'want', 'to', 'get', 'i', 'know', 'break', 'out', 'in', 'hives', 'almost', 'daily', 'and', 'have', 'to', 'take', 'zyrtec', 'like', 'it', 's', 'anyone', 'else', 'got', 'that', 'problem', 'can', 'you', 'expand', 'on', 'the', 'rare', 'complications', 'do', 'they', 'have', 'a', 'name', 'or', 'can', 'you', 'spell', 'out', 'symptoms', 'they', 'may', 'not', 'be', 'so', 'rare', 'if', 'people', 'knew', 'what', 'you', 'were', 'talking', 'about', 'thank', 'you', 'other', 'than', 'life', 'threatening', 'side', 'effects', 'what', 'does', 'the', 'vaccine', 'do', 'benefits', 'outweigh', 'the', 'effect', 'tell', 'that', 'to', 'the', 'people', 'who', 'got', 'it', 'smh', 'and', 'still', 'lining', 'up', 'for', 'what', 'is', 'literally', 'is', 'as', 'dangerous', 'as', 'a', 'cold', 'now', 'hahahahaha', 'i', 'bet', 'if', 'they', 'asked', 'for', 'a', 'refund', 'they', 'd', 'get', 'it', 'i', 'wondered', 'if', 'messing', 'with', 'the', 'immune', 'system', 'to', 'train', 'it', 'to', 'specific', 'protein', 'parts', 'could', 'lead', 'to', 'immune', 'responses', 'attacking', 'healthy', 'organs', 'etc', 'my', 'partner', 'has', 'developed', 'extreme', 'tiredness', 'mood', 'disorders', 'and', 'symptoms', 'almost', 'like', 'dementia', 'her', 'brother', 'had', 'a', 'brain', 'clot', 'her', 'sister', 'heart', 'failure', 'and', 'her', 'brother', 'in', 'law', 'extreme', 'chronic', 'joint', 'pain', 'inflammation', 'weight', 'loss', 'and', 'alzheimers', 'type', 'cognition', 'impairment', 'a', 'shell', 'of', 'the', 'man', 'he', 'was', 'all', 'coincidentally', 'since', 'covid', 'vaccination', 'i', 'had', 'the', 'vaccine', 'to', 'be', 'socially', 'compliant', 'travel', 'etc', 'my', 'joint', 'pain', 'post', 'vaccine', 'was', 'ongoing', 'and', 'so', 'didn', 't', 'take', 'boosters', 'an', 'orwellian', 'nightmare', 'time', 'you', 'were', 'a', 'social', 'pariah', 'and', 'outcast', 'otherwise', 'like', 'the', 'civil', 'war', 'it', 'even', 'divided', 'family', 'members', 'times', 'have', 'changed', 'and', 'never', 'will', 'get', 'better', 'didnt', 'study', 'me', 'how', 'many', 'undocumented', 'issues', 'are', 'there', 'millions', 'and', 'millions', 'recent', 'it', 'has', 'been', 'known', 'for', 'years', 'and', 'even', 'during', 'trials', 'that', 'using', 'mrna', 'spike', 'protein', 'was', 'having', 'problems', 'for', 'years', 'i', 'have', 'had', 'a', 'proven', 'vaccine', 'brain', 'injury', 'neuro', 'issues', 'that', 'have', 'changed', 'the', 'intelligence', 'and', 'natural', 'talents', 'i', 'was', 'born', 'with', 'the', 'life', 'i', 'had', 'is', 'gone', 'because', 'i', 'made', 'sure', 'i', 'did', 'my', 'patriotic', 'duty', 'and', 'got', 'the', 'first', 'issue', 'of', 'the', 'vaccine', 'and', 'booster', 'i', 'had', 'to', 'leave', 'a', 'great', 'career', 'and', 'for', 'months', 'this', 'year', 'was', 'bedridden', 'it', 'gets', 'better', 'and', 'then', 'it', 'get', 'real', 'bad', 'i', 'was', 'walking', 'steps', 'june', 'steps', 'in', 'august', 'before', 'i', 'had', 'to', 'sit', 'quickly', 'or', 'else', 'pass', 'out', 'but', 'get', 'this', 'i', 'do', 'not', 'qualify', 'for', 'disability', 'diabolical', 'if', 'we', 're', 'finally', 'hearing', 'about', 'it', 'in', 'spite', 'of', 'how', 'they', 'spin', 'it', 'then', 'it', 's', 'not', 'that', 'rare', 'now', 'wait', 'a', 'min', 'they', 'say', 'the', 'benefits', 'outweigh', 'the', 'riaks', 'but', 'what', 'are', 'the', 'benefits', 'they', 'already', 'said', 'in', 'the', 'past', 'that', 'it', 'doesnt', 'prevent', 'you', 'from', 'getting', 'it', 'and', 'it', 'doesnt', 'prevent', 'you', 'from', 'spreading', 'it', 'so', 'what', 'other', 'benefit', 'is', 'there', 'it', 'doesnt', 'provide', 'the', 'onky', 'two', 'things', 'its', 'supposed', 'to', 'do', 'on', 'the', 'may', 'i', 'had', 'my', 'first', 'and', 'only', 'vaccine', 'after', 'being', 'pressurised', 'by', 'the', 'government', 'doctors', 'scientists', 'friends', 'family', 'and', 'customers', 'due', 'to', 'my', 'job', 'so', 'left', 'arm', 'was', 'the', 'target', 'didn', 't', 'feel', 'the', 'best', 'the', 'next', 'day', 'could', 'definitely', 'tell', 'that', 'there', 'was', 'something', 'weird', 'in', 'my', 'body', 'a', 'week', 'and', 'a', 'half', 'days', 'later', 'i', 'started', 'to', 'develop', 'a', 'problem', 'with', 'my', 'left', 'hand', 'progressively', 'day', 'by', 'day', 'getting', 'worse', 'muscle', 'loss', 'fiery', 'hot', 'pins', 'and', 'needles', 'and', 'joint', 'pains', 'it', 'got', 'so', 'bad', 'i', 'went', 'to', 'ae', 'and', 'they', 'diagnosed', 'me', 'with', 'carpal', 'tunnel', 'syndrome', 'a', 'week', 'later', 'the', 'exact', 'same', 'thing', 'happened', 'to', 'my', 'right', 'hand', 'but', 'my', 'left', 'wrist', 'and', 'forearm', 'were', 'starting', 'to', 'get', 'it', 'basically', 'by', 'the', 'of', 'june', 'virtually', 'a', 'month', 'later', 'things', 'got', 'so', 'bad', 'that', 'i', 'couldn', 't', 'even', 'lift', 'my', 'arms', 'up', 'it', 'felt', 'like', 'i', 'was', 'being', 'paralysed', 'throughout', 'both', 'hands', 'and', 'arms', 'i', 'got', 'admitted', 'to', 'hospital', 'where', 'they', 'didn', 't', 'have', 'a', 'clue', 'what', 'was', 'happening', 'to', 'me', 'it', 'took', 'them', 'two', 'days', 'and', 'a', 'few', 'scans', 'later', 'then', 'they', 'drop', 'this', 'very', 'rare', 'auto', 'immune', 'disease', 'diagnosis', 'on', 'me', 'dermatomyositis', 'which', 'is', 'was', 'that', 'rare', 'that', 'they', 'didn', 't', 'know', 'much', 'about', 'it', 'and', 'there', 'wasn', 't', 'really', 'a', 'known', 'cure', 'for', 'it', 'then', 'they', 'tell', 'me', 'i', 'have', 'a', 'rare', 'blood', 'group', 'so', 'the', 'only', 'thing', 'that', 'helped', 'with', 'the', 'pain', 'was', 'steroids', 'quite', 'a', 'high', 'dose', 'and', 'a', 'bit', 'years', 'later', 'and', 'i', 'm', 'on', 'half', 'a', 'milligram', 'a', 'day', 'now', 'aswell', 'as', 'strong', 'immune', 'suppressant', 'medication', 'both', 'of', 'my', 'hands', 'and', 'wrists', 'have', 'never', 'been', 'the', 'same', 'since', 'my', 'right', 'hip', 'is', 'collapsing', 'and', 'i', 'now', 'have', 'a', 'blood', 'clot', 'in', 'the', 'same', 'leg', 'which', 'you', 'apparently', 'become', 'more', 'prone', 'to', 'with', 'this', 'disease', 'i', 'regret', 'every', 'day', 'that', 'i', 'went', 'and', 'had', 'it', 'done', 'i', 've', 'never', 'been', 'the', 'same', 'since', 'i', 'm', 'years', 'old', 'living', 'in', 'what', 'feels', 'like', 'a', 'year', 'olds', 'body', 'and', 'medication', 'dependent', 'i', 'know', 'something', 's', 'not', 'right', 'here', 'when', 'they', 'say', 'rare', 'they', 'are', 'not', 'giving', 'the', 'numbers', 'considering', 'that', 'covid', 'deaths', 'were', 'about', 'in', 'ten', 'thousand', 'one', 'could', 'also', 'say', 'the', 'vaccine', 'treated', 'a', 'rare', 'complication', 'from', 'covid', 'a', 'virus', 'so', 'deadly', 'that', 'you', 'needed', 'a', 'test', 'just', 'to', 'see', 'if', 'you', 'even', 'had', 'it', 'or', 'not', 'rare', 'how', 'rare', 'one', 'in', 'a', 'million', 'and', 'the', 'benefits', 'that', 'outweigh', 'the', 'risks', 'are', 'what', 'my', 'mom', 'and', 'i', 'got', 'three', 'shots', 'the', 'third', 'shot', 'i', 'was', 'fine', 'but', 'my', 'mom', 'got', 'a', 'blood', 'clot', 'in', 'her', 'foot', 'the', 'er', 'didn', 't', 'believe', 'her', 'until', 'she', 'forced', 'them', 'to', 'do', 'a', 'scan', 'they', 'had', 'to', 'go', 'in', 'and', 'get', 'rid', 'of', 'the', 'clot', 'because', 'my', 'mom', 'had', 'already', 'had', 'a', 'heart', 'attack', 'from', 'getting', 'the', 'virus', 'br', 'they', 'tell', 'you', 'to', 'get', 'it', 'if', 'you', 're', 'immune', 'compromised', 'or', 'otherwise', 'weak', 'against', 'illnesses', 'but', 'the', 'same', 'thing', 'that', 's', 'supposed', 'to', 'protect', 'you', 'can', 'easily', 'kill', 'you', 'br', 'we', 're', 'taking', 'our', 'chances', 'now', 'no', 'more', 'of', 'these', 'vaccines', 'the', 'ones', 'uses', 'for', 'years', 'and', 'well', 'studied', 'are', 'fine', 'the', 'new', 'ones', 'coming', 'out', 'with', 'all', 'of', 'these', 'issues', 'though', 'we', 'll', 'be', 'thinking', 'twice', 'before', 'trying', 'that', 'again', 'br', 'stop', 'using', 'people', 'as', 'guinea', 'pigs', 'my', 'mom', 'and', 'i', 'got', 'lucky', 'we', 'didn', 't', 'die', 'not', 'everyone', 'is', 'so', 'lucky', 'a', 'virus', 'that', 's', 'survivable', 'the', 'vaccine', 'outweighs', 'the', 'risks', 'of', 'brain', 'swelling', 'myocarditis', 'pericarditis', 'blood', 'clots', 'a', 'quot', 'vaccine', 'quot', 'that', 'doesn', 't', 'even', 'keep', 'you', 'from', 'catching', 'the', 'virus', 'who', 'the', 'hell', 'are', 'they', 'kidding', 'context', 'a', 'higher', 'percentage', 'of', 'people', 'suffer', 'a', 'rare', 'side', 'effect', 'from', 'drinking', 'water', 'than', 'this', 'one', 'so', 'what', 'are', 'the', 'odds', 'of', 'these', 'pfizer', 'sponsored', 'content', 'december', 'the', 'hospital', 'morticians', 'had', 'a', 'peer', 'reviewed', 'meeting', 'nationwide', 'pints', 'br', 'pints', 'br', 'pints', 'pints', 'br', 'pints', 'br', 'how', 'much', 'is', 'pint', 'br', 'pints', 'of', 'clots', 'on', 'these', 'yos', 'br', 'that', 'was', 'dec', 'why', 'for', 'not', 'only', 'covid', 'vaccine', 'but', 'many', 'other', 'medical', 'proceedures', 'have', 'i', 'never', 'nor', 'know', 'of', 'anyone', 'who', 'has', 'had', 'a', 'serious', 'adverse', 'reaction', 'if', 'adverse', 'reactions', 'were', 'as', 'serious', 'and', 'common', 'as', 'on', 'line', 'posters', 'claim', 'why', 'do', 'i', 'not', 'know', 'of', 'at', 'least', 'one', 'i', 'just', 'spoke', 'with', 'a', 'lady', 'proudly', 'telling', 'me', 'she', 'had', 'shots', 'now', 'they', 'give', 'covid', 'shots', 'in', 'one', 'arm', 'and', 'the', 'flue', 'shots', 'on', 'the', 'other', 'side', 'days', 'apart', 'crazy', 'putting', 'it', 'in', 'flu', 'and', 'shingle', 'shots', 'now', 'old', 'babies', 'too', 'those', 'with', 'the', 'side', 'effects', 'were', 'and', 'still', 'are', 'false', 'called', 'names', 'and', 'harassed', 'for', 'spreading', 'misinformatuion', 'people', 'will', 'get', 'what', 'they', 'deserve', 'benefits', 'equal', 'corporate', 'profits', 'of', 'billion', 'dollars', 'pureblood', 'and', 'proud', 'i', 'love', 'these', 'constantly', 'repeated', 'words', 'rare', 'exceptionally', 'rare', 'uncommon', 'side', 'effects', 'do', 'you', 'think', 'we', 'are', 'fools', 'this', 'is', 'only', 'tip', 'of', 'the', 'iceberg', 'and', 'the', 'vaccinated', 'are', 'still', 'g', 'e', 'tting', 'covid', 'lol', 'rare', 'hardly', 'what', 'benefit', 'i', 'assume', 'that', 'this', 'test', 'is', 'on', 'the', 'american', 'pfizer', 'vac', 'that', 'strangely', 'is', 'not', 'used', 'in', 'europe', 'why', 'oh', 'side', 'effects', 'who', 'would', 'guess', 'for', 'most', 'folks', 'under', 'who', 'are', 'generally', 'healthy', 'actually', 'getting', 'covid', 'presents', 'little', 'risks', 'and', 'there', 'are', 'effective', 'treatments', 'this', 'risk', 'is', 'even', 'lower', 'in', 'children', 'the', 'risk', 'of', 'significant', 'side', 'effects', 'is', 'much', 'higher', 'in', 'boys', 'and', 'young', 'men', 'so', 'measuring', 'the', 'balance', 'there', 'is', 'very', 'little', 'reason', 'to', 'give', 'the', 'vaccine', 'to', 'children', 'or', 'teens', 'unless', 'they', 'have', 'a', 'significant', 'side', 'issue', 'such', 'a', 'asmah', 'and', 'that', 'reason', 'gets', 'even', 'smaller', 'in', 'boys', 'and', 'young', 'men', 'then', 'look', 'at', 'the', 'benefits', 'side', 'the', 'vaccine', 'does', 'provide', 'enhanced', 'immunity', 'to', 'covid', 'in', 'most', 'recipients', 'for', 'a', 'limited', 'time', 'period', 'about', 'two', 'months', 'and', 'drops', 'off', 'very', 'rapidly', 'after', 'that', 'for', 'many', 'people', 'it', 'might', 'actually', 'be', 'better', 'to', 'actually', 'get', 'covid', 'that', 'in', 'most', 'cases', 'is', 'less', 'significant', 'than', 'a', 'case', 'of', 'the', 'flu', 'since', 'the', 'natural', 'immunity', 'that', 'produces', 'is', 'more', 'effective', 'and', 'longer', 'lasting', 'than', 'the', 'vaccine', 'immunity', 'older', 'folks', 'and', 'others', 'with', 'compromised', 'immunity', 'systems', 'probably', 'should', 'take', 'the', 'vaccine', 'and', 'mayb', 'stay', 'out', 'of', 'walmarts', 'and', 'movie', 'theatres', 'until', 'the', 'situation', 'passes', 'i', 'am', 'not', 'vaccinated', 'i', 'got', 'once', 'and', 'i', 'let', 'my', 'immune', 'system', 'to', 'deal', 'with', 'it', 'i', 'am', 'fully', 'recovered', 'afterwards', 'maybe', 'i', 'am', 'lucky', 'but', 'i', 'think', 'getting', 'the', 'shot', 'is', 'more', 'risky', 'than', 'getting', 'the', 'infection', 'i', 've', 'beaten', 'covid', 'twice', 'guys', 'once', 'with', 'monoclonal', 'antibodies', 'symptom', 'free', 'in', 'hours', 'and', 'once', 'with', 'the', 'much', 'vilified', 'ivermectin', 'symptom', 'free', 'in', 'less', 'than', 'hours', 'don', 't', 'believe', 'the', 'hype', 'given', 'to', 'these', 'bulls', 't', 'clot', 'shots', 'do', 'your', 'research', 'and', 'the', 'money', 'my', 'father', 'is', 'currently', 'in', 'the', 'hospital', 'with', 'multiple', 'undiagnosed', 'issues', 'he', 'got', 'the', 'shot', 'and', 'since', 'then', 'has', 'developed', 'skin', 'cancer', 'bladder', 'cancer', 'multiple', 'issues', 'of', 'urinary', 'tract', 'infection', 'all', 'the', 'symptoms', 'but', 'the', 'tests', 'come', 'back', 'negative', 'his', 'urine', 'literally', 'looks', 'like', 'dark', 'coffee', 'with', 'tons', 'of', 'sediment', 'at', 'the', 'bottom', 'consistent', 'equalibrium', 'issues', 'which', 'have', 'caused', 'falls', 'this', 'year', 'alone', 'he', 's', 'been', 'through', 'ct', 'amp', 'mri', 'scans', 'amp', 'sonigrams', 'of', 'the', 'legs', 'neck', 'back', 'kidneys', 'bladder', 'brain', 'lungs', 'and', 'heart', 'with', 'nothing', 'showing', 'up', 'yet', 'the', 'issues', 'are', 'there', 'and', 'very', 'prominent', 'bear', 'in', 'mind', 'my', 'father', 'is', 'years', 'old', 'but', 'before', 'the', 'shots', 'he', 'was', 'doing', 'miles', 'every', 'other', 'day', 'on', 'the', 'treadmill', 'and', 'minutes', 'on', 'an', 'elliptical', 'bike', 'his', 'doctors', 'told', 'him', 'he', 'would', 'live', 'to', 'if', 'he', 'kept', 'up', 'his', 'routines', 'then', 'the', 'shots', 'and', 'then', 'the', 'issues', 'which', 'have', 'been', 'for', 'years', 'after', 'pfizer', 'with', 'the', 'help', 'of', 'the', 'fda', 'tried', 'to', 'hide', 'their', 'data', 'for', 'years', 'through', 'the', 'court', 'system', 'i', 'm', 'convinced', 'these', 'poison', 'clot', 'creating', 'bulls', 't', 'shots', 'have', 'something', 'if', 'not', 'everything', 'too', 'with', 'his', 'issues', 'at', 'what', 'point', 'do', 'the', 'experts', 'begin', 'to', 'question', 'whether', 'the', 'benefits', 'outweigh', 'the', 'risks', 'how', 'many', 'other', 'complications', 'how', 'many', 'excess', 'deaths', 'benefits', 'haha', 'the', 'benefits', 'of', 'not', 'being', 'able', 'to', 'catch', 'or', 'transmit', 'covid', 'once', 'you', 'got', 'wait', 'it', 'didn', 't', 'stop', 'you', 'from', 'catching', 'or', 'it', 'a', 'vaccine', 'is', 'like', 'calling', 'a', 'steak', 'a', 'vegan', 'meal', 'sure', 'they', 'do', 'ginger', 'tea', 'lemon', 'honey', 'and', 'hot', 'water', 'steam', 'therapy', 'is', 'what', 'i', 'use', 'when', 'i', 'got', 'infected', 'with', 'covid', 'lot', 'of', 'hot', 'water', 'plain', 'green', 'tea', 'another', 'lie', 'the', 'benefits', 'do', 'not', 'vastly', 'outweigh', 'the', 'risks', 'especially', 'in', 'the', 'age', 'cohorts', 'the', 'officials', 'recommend', 'say', 'rare', 'a', 'few', 'more', 'times', 'maybe', 'they', 'will', 'believe', 'it', 'nope', 'time', 'will', 'tell', 'yo', 'me', 'la', 'puese', 'y', 'ni', 'me', 'puedo', 'lebantarme', 'de', 'la', 'cama', 'benefits', 'funny', 'how', 'most', 'conservatives', 'have', 'no', 'issues', 'with', 'any', 'vaccine', 'except', 'the', 'covid', 'one', 'they', 'just', 'reacts', 'as', 'their', 'orange', 'god', 'tells', 'them', 'to', 'its', 'worth', 'the', 'risk', 'take', 'your', 'booster', 'i', 'get', 'sick', 'after', 'being', 'around', 'the', 'newly', 'vaccinated', 'please', 'tell', 'our', 'leaders', 'to', 'stop', 'giving', 'these', 'oh', 'look', 'guys', 'all', 'of', 'a', 'sudden', 'there', 's', 'side', 'effects', 'thank', 'you', 'very', 'much', 'for', 'sharing', 'the', 'video', 'i', 'have', 'been', 'suffering', 'from', 'this', 'for', 'years', 'and', 'months', 'and', 'over', 'specialists', 'have', 'not', 'been', 'able', 'to', 'help', 'me', 'i', 'have', 'been', 'cleared', 'of', 'over', 'diseases', 'this', 'is', 'the', 'first', 'time', 'i', 'have', 'of', 'this', 'i', 'am', 'also', 'suffering', 'over', 'other', 'side', 'effects', 'and', 'all', 'the', 'doctors', 'do', 'not', 'know', 'why', 'largest', 'vaccine', 'safety', 'study', 'ever', 'your', 'first', 'question', 'should', 'be', 'why', 'wasn', 't', 'this', 'study', 'done', 'prior', 'to', 'people', 'receiving', 'the', 'shot', 'it', 's', 'rare', 'because', 'it', 's', 'rare', 'a', 'few', 'million', 'spikes', 'aren', 't', 'gon', 'na', 'do', 'more', 'than', 'several', 'billion', 'a', 'day', 'of', 'virus', 'spikes', 'always', 'cool', 'and', 'happy', 'to', 'blood', 'donation', 'daily', 'is', 'a', 'oxygen', 'to', 'consume', 'energy', 'light', 'energy', 'stocks', 'to', 'blood', 'clots', 'is', 'invisible', 'within', 'same', 'time', 'to', 'current', 'flows', 'in', 'light', 'invention', 'is', 'called', 'information', 'platform', 'when', 'you', 're', 'years', 'old', 'exercise', 'regularly', 'and', 'have', 'no', 'health', 'problems', 'the', 'risk', 'of', 'taking', 'the', 'vaccine', 'is', 'definitely', 'higher', 'the', 'fact', 'that', 'there', 'are', 'still', 'people', 'trying', 'to', 'gaslight', 'us', 'into', 'thinking', 'mass', 'vaccinations', 'were', 'a', 'net', 'good', 'is', 'really', 'scary', 'to', 'me', 'its', 'interesting', 'after', 'the', 'first', 'two', 'shots', 'my', 'health', 'started', 'to', 'decline', 'and', 'ive', 'suffered', 'injuries', 'aswell', 'after']
    [(('the', 'vaccine'), 14), (('of', 'the'), 12), (('side', 'effects'), 12), (('didn', 't'), 9), (('the', 'benefits'), 9), (('for', 'years'), 8), (('in', 'the'), 8), (('it', 's'), 8), (('i', 'have'), 7), (('outweigh', 'the'), 7), (('the', 'risks'), 7), (('i', 'was'), 7), (('and', 'i'), 7), (('i', 'am'), 7), (('the', 'disease'), 6), (('on', 'the'), 6), (('had', 'a'), 6), (('i', 'had'), 6), (('i', 'got'), 6), (('have', 'a'), 6)]



# **Discussion, Challenges, and Future Directions**

Both videos, on average, had 15% positive sentiment comments. The news video had more comments with negative sentiments compared to the explainer. The manual reading of the comments told a slightly different story. Contrary to my assumption, there were a significant number of anti-vaxxers in the comment section of the explainer video. Note that being in the comment section may or may not mean that they have actually watched the video. There was also hate between the two polarized communities seen in these comments.

As I read the comments, I also realized that sentiment analysis might not have been the best method for determining whether the commenter was an anti-vaxxer or not. First, there are junk comments that need to be removed. Second, an anti-vaxxer could say, "I'm glad I did not take the vaccine," which would be labeled as positive, whereas a comment by a pro-vaccine user, such as "the vaccine does not cause autism," gets labeled as negative. Therefore, there is a need to fine-tune transformer models in order for them to label the comments as positive, negative, or neutral in the context of vaccines and the content of the video. Or at the very least, the interpretations of the sentiments should be done in the context of the video.

From my limited review of the literature, I identified that YouTube comments have not been formally studied to understand how misinformation spreads. This only reinforces what Prof. Debayan mentioned in class about YouTube comments being an underexplored avenue in research.
