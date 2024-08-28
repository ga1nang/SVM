import os
import nltk
import time
import string
import operator
import numpy as np
import pandas as pd
from time import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')

#function for preprocessing text
def text_cleanup(text):
    text_without_punctuation = [c for c in text if c not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)
    text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    text_without_stopwords = ' '.join(text_without_stopwords)
    cleaned_text = [word.lower() for word in text_without_stopwords.split()]
    return cleaned_text


def prepare_dictionary():
    start_time = time()
    
    lmtzr = WordNetLemmatizer()
    k=0
    count = {}
    
    directory_in_str = 'data\\email'
    directory = os.fsencode(directory_in_str)
    
    for file in os.listdir(directory):
        file = file.decode("utf-8")
        file_name = str(os.getcwd()) + '\\data\\email\\'
        file_name = file_name + file
        print(file_name)
        file_reading = open(file_name, "r", encoding='utf-8', errors='ignore')
        words = text_cleanup(file_reading.read())
        for word in words:
            if (word.isdigit() == False and len(word) > 2):
                word = lmtzr.lemmatize(word)
                if word in count:
                    count[word] += 1
                else:
                    count[word] = 1
                    
        k += 1
        file_reading.close()
        if (k % 100 == 0):
            print("Done " + str(k))
            
    sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    sorted_count = dict(sorted_count)
    
    
    f = open("data\\wordslist.csv", "w+")
    f.write('word, count')
    f.write('\n')
    for word, times in sorted_count.items():
        if times < 100:
            break
        f.write(str(word) + ', ' + str(times))
        f.write('\n')
        
    f.close()
    
    print('Time (in seconds) to pre process the emails ' + str(round(time() - start_time,2)))
    
    
def prepare_data():
    start_time = time()
    
    df = pd.read_csv('data\\wordslist.csv', header=0)
    words = df['word']
    
    lmtzr = WordNetLemmatizer()
    
    directory_in_str = 'data\\email'
    directory = os.fsencode(directory_in_str)
    f = open('data\\frequency.csv', 'w+')
    for i in words:
        f.write(str(i) + ',')
    f.write('output')
    f.write('\n')
    f.close()
    
    k = 0
    
    for file in os.listdir(directory):
        file = file.decode('utf-8')
        file_name = str(os.getcwd()) + '\\data\\email\\'
        for i in file:
            if (i != 'b' and i != "'"):
                file_name = file_name + i
        k += 1
        file_reading = open(file_name, 'r', encoding='utf-8', errors='ignore')
        words_list_array = np.zeros(words.size)
        for word in file_reading.read().split():
            word = lmtzr.lemmatize(word.lower())
            if (word in stopwords.words('english') or word in string.punctuation or len(word) <= 2 or word.isdigit() == True):
                continue
            for i in range(words.size):
                if (words[i] == word):
                    words_list_array[i] = words_list_array[i] + 1
                    break
        f = open('data\\frequency.csv', 'a')
        for i in range(words.size):
            f.write(str(int(words_list_array[i])) + ',')
            
        print('file name', file_name, file_name.find('spam'))
        print(words_list_array)
        if (file_name.find('spam') > 0):
            f.write('1')
        elif (file_name.find('spam') == -1):
            f.write('0')
        f.write('\n')
        f.close()
        if (k % 100 == 0):
            print('Done' + str(k))
    
    print("Time (in seconds) to segregate entire dataset to form input vector " + str(round(time() - start_time,2)))
    
    
#main 
prepare_dictionary()
prepare_data()