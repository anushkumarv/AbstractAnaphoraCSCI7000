import nltk
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extractor_classifier import extract_features_classifier, load_data

data = '/Users/anushkumarv/Projects/AbstractAnaphoraCSCI7000/data/Resisting_rhetorics_of_language_endangerment/annotation.json'

print('Extract all features')
all_data = extract_features_classifier(load_data(data))
df = pd.DataFrame(all_data, columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
print('Train test split')
train, test = train_test_split(df, test_size=0.2, stratify=df['is_abs_anph'])
print(train[0])