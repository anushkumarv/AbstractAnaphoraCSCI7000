from typing import List
import json

from labels import get_lables
from helper import Helper

from tqdm import tqdm


label_non_abs_anph_set, label_non_abs_antd_set, label_abs_anph_set, label_abs_antd_set = get_lables()


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
            pronoun_path = helper.get_pronoun_path()


            # label_anph_obj = build_labels_dict.get(from_id).get("value")
            # label_antd_obj = build_labels_dict.get(to_id).get("value")
            # pronoun = pronoun_of_anph(build_labels_dict.get(from_id))
            # token_pos = relative_pos_anph_sent(datapoint.get("data").get("corefText"), label_anph_obj.get("start"), label_anph_obj.get("end"))
            # label_anph = label_anph_obj.get("labels")[0]
            # label_antd = label_antd_obj.get("labels")[0]
            # abs_anph = 1 if label_anph in label_abs_anph_set else 0
            # temp.append(label_anph)
            # temp.append(label_antd)
            # temp.append(abs_anph)
            data.append(temp)
        
        break

    return data



if __name__ == "__main__":
    label_std_data = load_data("/Users/anushkumarv/Projects/AbstractAnaphoraCSCI7000/data/Resisting_rhetorics_of_language_endangerment/annotation.json")
    print(parse_json(label_std_data))