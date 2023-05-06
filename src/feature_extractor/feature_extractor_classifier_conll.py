import pandas as pd

from helper_conll import get_filter_conll_data, HelperConllClassifier
from tqdm import tqdm

def extract_classifier_features():
    features = list()
    data = get_filter_conll_data()
    for key in tqdm(data):
        for i, coref_span in enumerate(data[key][1]):
            for j, item in enumerate(coref_span):
                if item[1] == item[2]:
                    anph = data[key][0][i][item[1]]
                    cluster = item[0]
                    if anph == 'This' or anph == 'this' or anph == 'That' or anph == 'that':
                        temp = list()
                        helper = HelperConllClassifier(data[key][0], i, item[1])
                        temp.append(anph.lower())
                        temp.append(item[1])
                        verb_presence = helper.get_verb_presence(); temp.append(verb_presence)
                        parent_lemma_verb = helper.get_parent_lemma(); temp.append(parent_lemma_verb)
                        parent_lemma, parent_label = helper.get_parent_and_label(); temp.append(parent_lemma); temp.append(parent_label)
                        negated_parent = helper.get_negated_parent(); temp.append(negated_parent)
                        parent_transitivity = helper.get_parent_transitivity(); temp.append(parent_transitivity)
                        pronoun_path = helper.get_pronoun_path(); temp.append(pronoun_path)
                        # abs_anph = helper.is_abstract_anaphora(); temp.append(abs_anph)
                        abs_anph = False
                        count = 3
                        for k in range(i-1, -1, -1):
                            if count > 0:
                                for prev_cluster, prev_start, prev_end in data[key][1][k]:
                                    if prev_cluster == cluster and prev_start < prev_end:
                                        abs_anph = helper.is_abstract_anaphora(data[key][0][k][prev_start:prev_end+1])
                                        if abs_anph:
                                            break
                            count -=1 
                            if abs_anph:
                                break
                        temp.append(abs_anph)
                        features.append(temp)
    return features


if __name__ == '__main__':
    data = extract_classifier_features()
    df = pd.DataFrame(data, columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
    print(df.head())
    print(len(df))
    print(len(df[df['is_abs_anph'] == True]))