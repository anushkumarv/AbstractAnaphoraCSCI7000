from typing import List
import json
import pandas as pd

from labels import get_lables
from helper import Helper

from tqdm import tqdm


def load_data(filename :str) -> List:
    with open(filename) as f:
        json_data = f.read()
    label_std_data = json.loads(json_data)
    return label_std_data


def parse_json(label_std_data :List):

    data = list()
    for datapoint in tqdm(label_std_data):
        annotaions = datapoint.get("annotations")[0].get("result")
        build_labels_dict = dict()
        build_relations_list = list()
        for label in annotaions:
            if not label.get("id") and label.get("type") == "relation":
                build_relations_list.append(label)
            else:
                build_labels_dict[label.get("id")] = label

        for relation in build_relations_list:
            temp = list()
            from_id = relation.get("from_id")
            to_id = relation.get("to_id")
            helper = Helper(build_labels_dict.get(from_id), datapoint, build_labels_dict.get(from_id).get("value"), build_labels_dict.get(to_id).get("value"))
            pronoun = helper.pronoun_of_anph()
            token_pos = helper.relative_pos_anph_sent()
            verb_presence = helper.get_verb_presence()
            parent_lemma_verb = helper.get_parent_lemma()
            parent_lemma, parent_label = helper.get_parent_and_label()
            negated_parent = helper.get_negated_parent()
            parent_transitivity = helper.get_parent_transitivity()
            pronoun_path = helper.get_pronoun_path()
            abs_anph = helper.is_abstract_anaphora()
            temp.append(pronoun)
            temp.append(token_pos)
            temp.append(verb_presence)
            temp.append(parent_lemma_verb)
            temp.append(parent_lemma)
            temp.append(parent_label)
            temp.append(negated_parent)
            temp.append(parent_transitivity)
            temp.append(pronoun_path)
            temp.append(abs_anph)
            data.append(temp)

    return data



if __name__ == "__main__":
    label_std_data = load_data("/Users/anushkumarv/Projects/AbstractAnaphoraCSCI7000/data/Resisting_rhetorics_of_language_endangerment/annotation.json")
    data = parse_json(label_std_data)
    df = pd.DataFrame(data, columns = ['pronoun', 'token_pos','verb_presence','parent_lemma_verb','parent_lemma','parent_label','negated_parent','parent_transitivity','pronoun_path','is_abs_anph'])
    print(df['parent_label'])
    print("This abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'This')]) + len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'this')]))
    print("This non abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'This')]) + len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'this')]))
    print("That abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'That')]) + len(df[(df['is_abs_anph'] == True) & (df['pronoun'] == 'that')]))
    print("That non abstract_anaphora")
    print(len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'That')]) + len(df[(df['is_abs_anph'] == False) & (df['pronoun'] == 'that')]))