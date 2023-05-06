import nltk
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extractor_classifier import extract_features_classifier, load_data
import collections
from nltk.metrics.scores import accuracy, precision, recall, f_measure

data = ['./../../data/Resisting_rhetorics_of_language_endangerment/annotation.json','./../../data/Talking_Indian/annotation.json','./../../data/Language_Ideology_Agency_and_Heritage_Language_Insecurity_across_Immigrant_Generations/annotation.json','./../../data/Kimberley_language_resourse_center/annotation.json']

print('Extract all features')
all_data = [extract_features_classifier(load_data(item)) for item in data]
df = pd.DataFrame(columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
for item in all_data:
    df_temp = pd.DataFrame(item, columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
    df = df.append(df_temp)

print('Train test split')
train, test = train_test_split(df, test_size=0.2, stratify=df['is_abs_anph'])

y_train, x_train = train['is_abs_anph'], train.drop('is_abs_anph', axis=1)
y_test, x_test = test['is_abs_anph'], test.drop('is_abs_anph', axis=1)
x_train_lst  = x_train.to_dict('records')
y_train_lst = y_train.to_list()
x_test_lst = x_test.to_dict('records')
y_test_lst = y_test.to_list()
nltk_train = [(feature_train, target) for feature_train, target in zip(x_train_lst, y_train_lst)]

print('Train Navie Bayes Classifier')
classifier_nb = nltk.classify.NaiveBayesClassifier.train(nltk_train)
results_nb = classifier_nb.classify_many(x_test_lst)
print(results_nb)

print('Train Max Ent Classifier')
classifier_me = nltk.classify.MaxentClassifier.train(nltk_train, 'IIS', trace=0)
results_me = classifier_me.classify_many(x_test_lst)
print(results_me)


print("Length of training dataset", len(train))
print("Length of positive datapoints", len(train[train['is_abs_anph'] == True]))
print("Percentage of positive datapoints", len(train[train['is_abs_anph'] == True]) / len(train))

print("## Naive Bayes ## Accuracy on test set", accuracy(y_test_lst, results_nb))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (ref, pred) in enumerate(zip(y_test_lst, results_nb)):
    refsets[ref].add(i)
    testsets[pred].add(i)

print("## Naive Bayes ## precision on True label", precision(refsets[True], testsets[True]))
print("## Naive Bayes ## recall on True label", recall(refsets[True], testsets[True]))
print("## Naive Bayes ## f measure on True label", f_measure(refsets[True], testsets[True]))

print("## Naive Bayes ## precision on False label", precision(refsets[False], testsets[False]))
print("## Naive Bayes ## recall on False label", recall(refsets[False], testsets[False]))
print("## Naive Bayes ## f measure on False label", f_measure(refsets[False], testsets[False]))


print("## Max Ent ## Accuracy on test set", accuracy(y_test_lst, results_me))

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (ref, pred) in enumerate(zip(y_test_lst, results_nb)):
    refsets[ref].add(i)
    testsets[pred].add(i)

print("## Max Ent ## precision on True label", precision(refsets[True], testsets[True]))
print("## Max Ent ## recall on True label", recall(refsets[True], testsets[True]))
print("## Max Ent ## f measure on True label", f_measure(refsets[True], testsets[True]))

print("## Max Ent ## precision on False label", precision(refsets[False], testsets[False]))
print("## Max Ent ## recall on False label", recall(refsets[False], testsets[False]))
print("## Max Ent ## f measure on False label", f_measure(refsets[False], testsets[False]))