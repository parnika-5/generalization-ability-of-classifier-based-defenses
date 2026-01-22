# generalization-ability-of-classifier-based-defenses
quantitatively measure how well classifier defenses perform on known vs. unseen attacks.  We compare two pipelines: A TF-IDF + Logistic Regression classifier   A BERT-based semantic classifier   And our focus is specifically on the F1 performance drop between familiar and unfamiliar attacks.

README
To replicate our results no prior dependencies need to be downloaded. All required dependencies are embedded in the code and can be run as an ipynb notebook. 

We preload these dependencies before running our code: 

!pip install datasets transformers scikit-learn pandas


!pip install transformers datasets scikit-learn torch pandas numpy matplotlib seaborn -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import torch
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import warnings

Additionally, we use these 4 datasets to conduct our experiment: 

   1. deepset/prompt-injections - Standard injection examples
    2. Harelix/Prompt-Injection-Mixed-Techniques-2024 - Advanced attack methods
    3. jackhhao/jailbreak-classification - Jailbreak patterns (unseen category)
    4. alespalla/chatbot_instruction_prompts - Normal user prompts

