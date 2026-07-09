'''
MB: This is the sample code from the BERTopic website with some
placeholder code for loading StackOverflow data. This currently
does both fitting and prediction, but we can split that into separate tasks.  
'''

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd
from bs4 import BeautifulSoup
import xmltodict
import time
from datetime import datetime

'''
Download data from [INSERT LINK], save to preferred location, reference 
'Comments.xml' data location in data load below.
'''

with open('data_file', 'r') as f:
    data = f.read()

xml_dict = xmltodict.parse(data)

'''
Structure of xml file contents is {'comments': {'row': [{}, {}, {}, ...]}}. Value of 'row' dict is a list of dicts with the following key, value pairs:

'@Id', str(int) representing unique post identifier?
'@PostId', str(int) representing unique post identifier?
'@Score', str(int) representing number of up-votes received?
'@Text', str representing comment text
'@CreationDate', str representing timestamp yyyy-mm-ddThh:mm:ss.sss
'@UserId' OR '@UserDisplayName', str(int) representing unique user identifier?
'@ContentLicense', str representing name of content license for comment

This dataset contains 87,720 Astronomy Stack Exchange comments made between September 24, 2013 and 
December 31, 2025, with 6,795 unique users represented as commenters.'''

# Learn about users
unique_userids = {}

for n in xml_dict['comments']['row']:
    for k,v in n.items():
        if k == '@UserId':
            if v not in unique_userids:
                unique_userids[v]=1
            else:
                unique_userids[v]+=1

        elif k == '@UserDisplayName':
            if v not in unique_userids:
                unique_userids[v]=0
                unique_userids[v]+=1
            else:
                unique_userids[v]+=1
        else:
            continue



df = pd.read_csv('stackoverflow_comments.csv')
docs = df['text'].tolist()

# Load a custom embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pass it into BERTopic
topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(docs)
