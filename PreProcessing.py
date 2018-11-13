from __future__ import unicode_literals
import string
import pandas as pd
import codecs
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
pd.set_option('display.width', 2000)


class PreProcessing:
    Data = None
    Language = None

    def __init__(self,Data,language):
        self.Data = Data
        self.Language = language

    def Remove_Punctuation(self):
        for i in range(len(self.Data)):
            try:
                mess = self.Data[i]
                translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
                self.Data[i] = mess.translate(translator)
            except:
                print('in except')
                pass
        return self.Data

    def Remove_StopWords(self):
        stopwords_persian = '../stopwords-persian'
        stopwords_set = set(stopwords.words(self.Language))
        for i in range(len(self.Data)):
            try:
                words = word_tokenize(self.Data[i])
                wordsFiltered = [word for word in words if word not in stopwords_set]
                wordsFiltered = ' '.join(wordsFiltered)
                self.Data[i] = wordsFiltered
            except:
                pass
        return self.Data
