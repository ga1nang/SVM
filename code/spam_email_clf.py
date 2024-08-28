import string
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


#load and prepare data
df1 = pd.read_csv('data\\wordslist.csv')
df2 = pd.read_csv('data\\frequency.csv', header=0)

input_output = df2.values
X = input_output[:, :-1]
y = input_output[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)


#prepare SVC model
classifier = SVC(C = 100, kernel='linear')
classifier.fit(X_train, np.ravel(y_train))


#function preprocess email before predict
words = df1['word']
lmtzr = WordNetLemmatizer()

def encode_email(fileName, words):
    file_reading = open(fileName, 'r', encoding='utf-8', errors='ignore')
    words_list_array = np.zeros(words.size)
    for word in file_reading.read().split():
        word = lmtzr.lemmatize(word.lower())
        
        if(word in stopwords.words('english') or word in string.punctuation or len(word)<=2 or word.isdigit()==True):
            continue
        
        for i in range(words.size):
            if (words[i] == word):
                words_list_array[i] = words_list_array[i] + 1
                break
            
    return words_list_array


def check_email(fileName, words, classifier):
    email = encode_email(fileName, words)
    if (classifier.predict([email])[0] == 1):
        print('Spam email')
    else:
        print('Normal email')
        

result1 = check_email('data/email/0028.1999-12-17.farmer.ham.txt', words, classifier)
result2 = check_email('data/email/0334.2004-01-30.GP.spam.txt', words, classifier)      
      
