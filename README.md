# Jr Data Scientist - Evaluation -1 

The code for the 3 tasks are present in the task.ipynb file  

I have used happytransformer, vaderSentiment, summa and language-check libraries which can be installed by running the first cell in the task.ipynb file  

For Task 2 Sentiment Mismatch I have extracted those reviews that sound positive with low stars and those that sounf negative with High stars.  

In Part 2 of this task I studied study the correlation of the Rankings I have extracted top 15 keywords and saw correlation between how many times these 15 
words are present in the first 10 words of the Description and the Rank. The code and correlation can be found under `Ranking Dataset` Section.  

I had tough time Doing Grammar check because of the limited compute resource so I wrote a Multiprocessing code in multiprocessed_grammar.py where with 5 workers
I was able to generate the results.  

I have also used Transformer based model to produce Grammar corrected text when we provide it an Grammatically incorrect text.  

Finally I created a Flask Web App and hosted it on GCP. There is a functionality to ulpoad a reviews dataset and see the grammatically incorrect reviews. 
