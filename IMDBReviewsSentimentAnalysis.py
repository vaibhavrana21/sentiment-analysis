import nltk, csv
from nltk.sentiment import SentimentIntensityAnalyzer
from pattern.text.en import sentiment
from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob

classifier = TextClassifier.load('en-sentiment')


nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

positive = negative = neutral = correct = 0

with open("C:\\DataScience\IMDBReviews.csv", 'r', encoding='utf-8') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    #print(row[0])
    review = row[0].replace('<br>','')

    #sentimentVader = sia.polarity_scores(review)['compound']

    #sentimentPattern = float(sentiment(review))

    #sentimentTB = TextBlob(review).sentiment[0]

    sentence = Sentence(review)
    classifier.predict(sentence)
    sentimentFlair = str(sentence.labels)[-20:]
    '''
    if sentimentTB>0:
        result = "positive"
        positive = positive+1

    elif sentimentTB<0:
        result = "negative"
        negative = negative+1
    else:
        result = "neutral"
        neutral = neutral+1
    if result == row[1]:
        correct = correct + 1
    '''
    
    if "POSITIVE" in sentimentFlair:
        result = "positive"
        positive = positive+1

    elif "NEGATIVE" in sentimentFlair:
        result = "negative"
        negative = negative+1
    else:
        result = "neutral"
        neutral = neutral+1
    if result == row[1]:
        correct = correct + 1


print("Positive: " + positive)
print("Negative: " + negative)
print("Neutral: " + neutral)
print(str(correct/(positive+negative+neutral)+"% accuracy")

