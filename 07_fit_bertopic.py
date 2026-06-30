'''
MB: This is the sample code from the BERTopic website with some
placeholder code for loading StackOverflow data. This currently
does both fitting and prediction, but we can split that into separate tasks.  
'''

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd

df = pd.read_csv('stackoverflow_comments.csv')
docs = df['text'].tolist()

# Load a custom embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pass it into BERTopic
topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(docs)
