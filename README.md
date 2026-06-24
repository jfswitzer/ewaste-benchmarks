# ewaste-benchmarks
This repository contains code for benchmarks to be tested on a cell phone e-waste server prototype.

# Benchmark 1 - Privacy Preservation: Synthetic data generation (or inference) using Llama LLM. 

# Benchmark 2 - Statistical Analysis: Understand relationships between variables by training computationally intensive random effects model 

# Benchmark 3 – Statistical Analysis: Identify relationships between variables by training simple logistic regression model 
Logistic Regression models are commonly used by social science researchers to understand statistical relationships among variables in a dataset. This benchmark reproduces Hemphill et al.’s (2020a, 2020b) logistic regression model training task. Code for this task is available through Hemphill et al.'s public GitHub repository here: https://github.com/casmlab/modeling-political-attention.git. We use code from the file pathway notebooks/models/create/supervised/1.0-ams-create-best-classifiers.ipynb, specifically code under the 'Make Training and Test Sets' heading until the end of the notebook.

# Benchmark 4 – Statistical Analysis: Infer classifications according to pre-trained logistic regression model 
Logistic Regression models are commonly used by social science researchers to understand statistical relationships among variables in a dataset. This benchmark reproduces Hemphill et al.’s (2020a, 2020b) logistic regression model inference task. Code for this task is available through Hemphill et al.'s public GitHub repository here: https://github.com/casmlab/modeling-political-attention.git. We use code from the file pathway notebooks/models/run/1.0-ams-label-comparisons.ipynb, specifically code under the 'Label Data with Supervised Model Predictions' heading.

# Benchmark 5 – Topic Modeling: Identify common thematic patterns across datapoints by training LDA model
LDA topic models are often used by social science researchers to understand thematic patterns across large data corpora. This benchmark reproduces Hemphill et al.’s (2020a, 2020b) LDA model training task.  Code for this task is available through Hemphill et al.'s public GitHub repository here: https://github.com/casmlab/modeling-political-attention.git. We use code from the file pathway notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET, specifically code under the section 'Command Line Model Training' heading.

# Benchmark 6 - Topic Modeling: Infer topics in a dataset according to pre-trained LDA model 
LDA topic models are often used by social science researchers to understand thematic patterns across large data corpora. This benchmark reproduces Hemphill et al.’s (2020a, 2020b) LDA model inference task. Code for this task is available through Hemphill et al.'s public GitHub repository here: https://github.com/casmlab/modeling-political-attention.git. We use code from the file pathway notebooks/models/create/unsupervised/1.0-ams-unigram-lda-model-MALLET, specifically code under the section 'Command Line Topic Inference' heading.

# Benchmark 7 - Topic Modeling: Identify common thematic patterns across datapoints by training BERTopic model

# Benchmark 8 - Topic Modeling: Infer topics in a dataset according to pre-trained BERTopic model
