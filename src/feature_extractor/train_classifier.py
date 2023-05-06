import argparse
import nltk
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extractor_classifier import extract_features_classifier, load_data
from report_results_helper import report_results
from feature_extractor_classifier_conll import extract_classifier_features as extract_classifier_features_conll

def train_and_report(args):
    data = ['./../../data/Resisting_rhetorics_of_language_endangerment/annotation.json','./../../data/Talking_Indian/annotation.json','./../../data/Language_Ideology_Agency_and_Heritage_Language_Insecurity_across_Immigrant_Generations/annotation.json','./../../data/Kimberley_language_resourse_center/annotation.json']

    print('Extract all features')
    all_data = [extract_features_classifier(load_data(item)) for item in data]
    df = pd.DataFrame(columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
    for item in all_data:
        df_temp = pd.DataFrame(item, columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
        df = df.append(df_temp)

    if args.use_conll == "True":
        print("Extract features from conll")
        df_conll = pd.DataFrame(extract_classifier_features_conll(), columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
        df = df.append(df_conll)

    print('Train test split')
    train, test = train_test_split(df, test_size=0.2, stratify=df['is_abs_anph'], random_state=25)

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

    print("This abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'This')]) + len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'this')]))
    print("This non abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'This')]) + len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'this')]))
    print("That abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'That')]) + len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'that')]))
    print("That non abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'That')]) + len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'that')]))

    print("Length of training dataset", len(train))
    print("Length of positive datapoints", len(train[train['is_abs_anph'] == True]))
    print("Percentage of positive datapoints", len(train[train['is_abs_anph'] == True]) / len(train))
    print("Length of test dataset", len(test))


    report_results("## Naive Bayes Classifier ##", y_test_lst, results_nb)
    report_results("## Max Ent Classifier ##", y_test_lst, results_me)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_conll', type=str, required=True)
    args = parser.parse_args()
    train_and_report(args)