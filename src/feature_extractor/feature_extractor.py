from typing import List
import json

from labels import get_lables


label_non_abs_anph_set, label_non_abs_antd_set, label_abs_anph_set, label_abs_antd_set = get_lables()


def load_data(filename :str) -> List:
    with open(filename) as f:
        json_data = f.read()
    label_std_data = json.loads(json_data)
    return label_std_data


def parse_json(label_std_data :List):

    data = list()
    for datapoint in label_std_data:
        annotaions = datapoint.get("annotations")[0].get("result")
        build_labels_dict = dict()
        for label in annotaions:
            build_labels_dict[label.get("id")] = label
        for key in build_labels_dict:
            if build_labels_dict.get(key).get("type") == "relation":
                from_id = build_labels_dict.get(key).get("from_id")
                label_anph = build_labels_dict.get(from_id).get("value").get("labels")[0]
                if label_anph in label_abs_anph_set:
                    temp = list()
                    to_id = build_labels_dict.get(key).get("to_id")
                    label_antd = build_labels_dict.get(to_id).get("value").get("labels")[0]
                    temp.append(label_anph)
                    temp.append(label_antd)
                
                    data.append(temp)

    return data



if __name__ == "__main__":
    label_std_data = load_data("../../data/Resisting_rhetorics_of_language_endangerment/annotation.json")
    print(parse_json(label_std_data))