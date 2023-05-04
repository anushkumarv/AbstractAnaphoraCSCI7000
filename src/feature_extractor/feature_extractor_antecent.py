from typing import List
from tqdm import tqdm
import pandas as pd

from helper import load_data, HelperAntecedent


# def feature_extractor_antecedents(label_std_data :List) -> List:
#     data = list()
#     for datapoint in tqdm(label_std_data):
#         annotaions = datapoint.get("annotations")[0].get("result")
#         build_labels_dict = dict()
#         build_relations_list = list()
#         for label in annotaions:
#             if not label.get("id") and label.get("type") == "relation":
#                 build_relations_list.append(label)
#             else:
#                 build_labels_dict[label.get("id")] = label

#         for relation in build_relations_list:
#             temp = list()
#             from_id = relation.get("from_id")
#             to_id = relation.get("to_id")
#             helper = HelperAntecedent(build_labels_dict.get(from_id), datapoint, build_labels_dict.get(from_id).get("value"), build_labels_dict.get(to_id).get("value"))
#             sent_distance = helper.get_sent_distance(); temp.append(sent_distance)
#             token_distance = helper.get_log_token_distance(); temp.append(token_distance)
#             relative_pos = helper.get_relative_positon(); temp.append(relative_pos)
#             direct_dominance = helper.get_direct_dominance(); temp.append(direct_dominance)
#             dominance = helper.get_dominance(); temp.append(dominance)
#             candidate_path = helper.get_candidate_path(); temp.append(candidate_path)
#             negated_candidate = helper.get_negated_candidate(); temp.append(negated_candidate)
#             candidate_transitivity = helper.get_candidate_transitivity(); temp.append(candidate_transitivity)
#             abs_anph = helper.is_abstract_anaphora(); temp.append(abs_anph) 
#             data.append(temp)

#     return data


def feature_extractor_antecedents(label_std_data :List) -> List:
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
            from_id = relation.get("from_id")
            to_id = relation.get("to_id")
            helper = HelperAntecedent(build_labels_dict.get(from_id), datapoint, build_labels_dict.get(from_id).get("value"), build_labels_dict.get(to_id).get("value"))
            candidates = helper.get_verb_candidates()
            for i, j in candidates:
                temp = list()
                sent_distance = helper.get_sent_distance(i); temp.append(sent_distance)
                token_distance = helper.get_log_token_distance(i, j); temp.append(token_distance)
                relative_pos = helper.get_relative_positon(i, j); temp.append(relative_pos)
                direct_dominance = helper.get_direct_dominance(i, j); temp.append(direct_dominance)
                dominance = helper.get_dominance(i, j); temp.append(dominance)
                candidate_path = helper.get_candidate_path(i, j); temp.append(candidate_path)
                negated_candidate = helper.get_negated_candidate(i, j); temp.append(negated_candidate)
                candidate_transitivity = helper.get_candidate_transitivity(i, j); temp.append(candidate_transitivity)
                is_antd = helper.is_antd(i, j); temp.append(is_antd)
                data.append(temp)

    return data

if __name__ == '__main__':
    label_std_data = load_data("/Users/anushkumarv/Projects/AbstractAnaphoraCSCI7000/data/Resisting_rhetorics_of_language_endangerment/annotation.json")
    data = feature_extractor_antecedents(label_std_data)
    df = pd.DataFrame(data, columns = ['sent_distance', 'token_distance','relative_pos','direct_dominance','dominance','candidate_path','negated_candidate','candidate_transitivity','is_antd'])
    print(df)

