import pandas as pd

from helper_conll import get_filter_conll_data, HelperConllAntecedent
from tqdm import tqdm


def extract_antecedent_features():
    features = list()
    data = get_filter_conll_data()
    for key in tqdm(data):
        for i, coref_span in enumerate(data[key][1]):
            for j, item in enumerate(coref_span):
                if item[1] == item[2]:
                    anph = data[key][0][i][item[1]]
                    cluster = item[0]
                    if anph == 'This' or anph == 'this' or anph == 'That' or anph == 'that':
                        count = 3
                        helper = HelperConllAntecedent(data[key][0], i, item[1])
                        abs_anph = False
                        for k in range(i, -1, -1):
                            if count >= 0:
                                for prev_cluster, prev_start, prev_end in data[key][1][k]:
                                    if prev_cluster == cluster and prev_start < prev_end:
                                        abs_anph = helper.is_abstract_anaphora(data[key][0][k][prev_start:prev_end+1])
                                        if abs_anph:
                                            candidates = helper.get_verb_candidates()
                                            for sent_id, cand_vb_id in candidates:
                                                temp = list()
                                                sent_distance = helper.get_sent_distance(sent_id); temp.append(sent_distance)
                                                token_distance = helper.get_log_token_distance(sent_id, cand_vb_id); temp.append(token_distance)
                                                relative_pos = helper.get_relative_positon(sent_id, cand_vb_id); temp.append(relative_pos)
                                                direct_dominance = helper.get_direct_dominance(sent_id, cand_vb_id); temp.append(direct_dominance)
                                                dominance = helper.get_dominance(sent_id, cand_vb_id); temp.append(dominance)
                                                candidate_path = helper.get_candidate_path(sent_id, cand_vb_id); temp.append(candidate_path)
                                                negated_candidate = helper.get_negated_candidate(sent_id, cand_vb_id); temp.append(negated_candidate)
                                                candidate_transitivity = helper.get_candidate_transitivity(sent_id, cand_vb_id); temp.append(candidate_transitivity)
                                                is_antd = helper.is_antd(sent_id, cand_vb_id, k, prev_start, prev_end); temp.append(is_antd)
                                                features.append(temp)
                            count -=1 
                            if abs_anph:
                                break
    return features



if __name__ == '__main__':
    data = extract_antecedent_features()
    df = pd.DataFrame(data, columns = ['sent_distance', 'token_distance','relative_pos','direct_dominance','dominance','candidate_path','negated_candidate','candidate_transitivity','is_antd'])
    print(df.head())
    print(len(df))