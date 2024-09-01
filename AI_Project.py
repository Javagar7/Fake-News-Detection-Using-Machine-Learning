import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier

#Importing the data set
print('Importing the data set')
fake = pd.read_csv("Fake.csv")
genuine = pd.read_csv("True.csv")

fake['target'] = 0
genuine['target'] = 1
data=pd.concat([fake,genuine],axis=0)

data=data.reset_index(drop=True)

data=data.drop(['subject','date','title'],axis=1)

#Data Preprocessing
#Tokenizing
print('Tokenizing')
data['text']=data['text'].apply(word_tokenize)

#Stemming
print('Stemming')
porter=SnowballStemmer("english",ignore_stopwords=False)
def stem_it(text):
    return [porter.stem(word) for word in text]
data['text']=data['text'].apply(stem_it)

#Stopword Removal
print('Stopword Removal')
def stop_it(t):
    dt=[word for word in t if len(word)>2]
    return dt
data['text']=data['text'].apply(stop_it)
data['text']=data['text'].apply(' '.join)

'''print(X_train.head())
print("\n")
print(y_train.head())'''

#Splitting dataset
print('Splitting dataset')
X_train,X_test,y_train,y_test = train_test_split(data['text'],data['target'],test_size=0.25)

#Vectorization (TFIDF)
print('Vectorization (TFIDF)')
my_tfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)

'''print(tfidf_train)'''

#Building of ML model
print('Building of ML model')
model_1 = LogisticRegression(max_iter = 900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1 = accuracy_score(y_test,pred_1)

print('The accuracy prediction in Passive Model is ',cr1*100)
print('Building of ML model')
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)

y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test,y_pred)

print('The accuracy prediction in Aggressive Model is ',accscore*100)

