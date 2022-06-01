import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from summa import keywords
import language_tool_python
import warnings
from concurrent.futures import ProcessPoolExecutor
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

tool = language_tool_python.LanguageTool('en-US')

def check_filter(string):
        methods = tool.check(string)
        value = len(methods)
        return min(value,1)

def main(text, results):
    with ProcessPoolExecutor(max_workers=5) as executor:
        for words, gram in zip(text, executor.map(check_filter, text)):
            results.append((words, gram))

    grammar = pd.DataFrame({'text':[i[0] for i in results], 'is_Error':[i[1] for i in results]})
    grammar.to_csv('./grammar_results.csv',index=False)

if __name__ == '__main__':
    review = pd.read_csv('review_data.csv')
    print(review.shape)
    text = review.text.values.tolist()
    results = []
    main(text, results)