# Import libraries
import pandas as pd 
import numpy as np
import nltk
import re
import pickle
import unicodedata
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

# Functions

stop=stopwords.words('english')
def remove_stopwords(message):
  if message is not None:
    words = message.strip().split()
    words_filtered = []
    for word in words:
      if word not in stop:
        words_filtered.append(word) 
    result = " ".join(words_filtered)         
  else:
    result = None

  return result  

def normalize_string(message):
  if message is not None:
     result = unicodedata.normalize('NFC',message).encode('ascii','ignore').decode()
  else:
    result = None
  return result  

def replace_multiple_letters(message):
  if message is not None:
    result = re.sub(r"([a-zA-Z])\1{2,}",r"\1",message)
  else:
    result= None
  return result  

# Load the data
df_raw = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/spam.csv")

# Clean the data
df_raw['Message'] = df_raw['Message'].str.lower()
df_raw['Message'].str.split(expand=True).stack().value_counts()[:60]
df_interin = df_raw.copy()
df_interin['Message'] = df_interin['Message'].apply(remove_stopwords)
df_interin['Message'] = df_interin['Message'].str.replace(".","",regex=False)
df_interin['Message'] = df_interin['Message'].str.replace('''[?&#,;ü']''','',regex=True)
df_interin['Message']= df_interin['Message'].apply(normalize_string)
df_interin['Message']= df_interin['Message'].apply(replace_multiple_letters)
df = df_interin.copy()

# Separate target from feature
X = df['Message']
y = df['Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=121)

# Vectorizador
vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

# Make the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
filename = '../models/NLP_Model.sav'
pickle.dump(model, open(filename, 'wb'))

print(model.score(X_test,y_test))


