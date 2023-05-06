import json
import textacy
import spacy
import math
from nltk.tokenize import word_tokenize
from typing import Union, List

from labels import get_lables

nlp = spacy.load("en_core_web_sm")

label_non_abs_anph_set, label_non_abs_antd_set, label_abs_anph_set, label_abs_antd_set = get_lables()

def load_data(filename :str) -> List:
    with open(filename) as f:
        json_data = f.read()
    label_std_data = json.loads(json_data)
    return label_std_data


class HelperClassifier:
    def __init__(self, label: dict, datapoint: dict, label_anph_obj: dict, label_antd_obj: dict) -> None:
        self.label = label
        self.datapoint = datapoint
        self.label_anph_obj = label_anph_obj
        self.label_antd_obj = label_antd_obj
        self.data = self.datapoint.get("data").get("corefText")
        self.anph_start_pos = self.label_anph_obj.get("start")
        self.anph_end_pos = self.label_anph_obj.get("end")
        self.doc = nlp(datapoint.get("data").get("corefText"))
        self.sents = list(self.doc.sents)
        self.anph_sent_id = self.get_anph_sent_id()

    def pronoun_of_anph(self) -> str:
        return word_tokenize(self.label.get("value").get("text"))[0]

    def get_anph_sent_id(self) -> int:
        start = 0
        for i in range(len(self.sents)):
            end = start + len(str(self.sents[i]))
            if self.anph_start_pos >= start  and self.anph_start_pos <= end:
                return i
            start = end + 1
        return len(self.sents) - 1


    def start_idx_anph_relative(self) -> int:
        start = sum([len(str(self.sents[i])) + 1 for i in range(self.anph_sent_id)])
        return self.anph_start_pos - start


    def relative_pos_anph_sent(self) -> int:
        anph_sent = str(self.sents[self.anph_sent_id])
        relative_idx = self.start_idx_anph_relative()
        anph = self.pronoun_of_anph()
        for i, token in enumerate(self.sents[self.anph_sent_id]):
            if token.text == anph:
                return i + 1
        return 0
        
    def get_verb_presence(self) -> int:
        patterns = [{"POS": "VERB"}]
        count = 0
        for i in range(self.anph_sent_id -1, -1, -1):
            verbs = textacy.extract.token_matches(self.sents[i], patterns=patterns)
            if sum(1 for _ in verbs) > 0:
                count += 1
        return count
    
    def get_pronoun_path(self) -> str:
        anph = self.pronoun_of_anph()
        for token in self.sents[self.anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if str(child) == anph:
                        return child.dep_
        return ''
    
    def get_parent_lemma(self) -> str:
        anph = self.pronoun_of_anph()
        for token in self.sents[self.anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.text == anph:
                        return token.lemma_
            
        return ''
    
    def get_parent_and_label(self) -> Union[str, str]:
        anph = self.pronoun_of_anph()
        for token in self.sents[self.anph_sent_id]:
            if token.text == anph and token.has_head:
                return token.head.lemma_,token.dep_
        return '', ''    
    
    def get_negated_parent(self) -> bool:
        anph = self.pronoun_of_anph()
        for token in self.sents[self.anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.dep_ == 'neg':
                        return True
        return False
    
    def get_parent_transitivity(self) -> bool:
        anph = self.pronoun_of_anph()
        for token in self.sents[self.anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if "obj" in child.dep_:
                        return True
        
        return False
    
    def is_abstract_anaphora(self) -> bool:
        return True if self.label_anph_obj.get("labels")[0] in label_abs_anph_set else False
    

# class HelperAntecedent(HelperClassifier):
#     def __init__(self, label: dict, datapoint: dict, label_anph_obj: dict, label_antd_obj: dict) -> None:
#         super().__init__(label, datapoint, label_anph_obj, label_antd_obj)
#         self.antd_start_pos = self.label_antd_obj.get("start")
#         self.antd_end_pos = self.label_antd_obj.get("end")
#         self.antd = self.data[self.antd_start_pos: self.antd_end_pos+1]
#         self.antd_sent_id = self.get_antd_sent_id()

#     def get_antd_sent_id(self) -> int:
#         start = 0
#         for i in range(len(self.sents)):
#             end = start + len(str(self.sents[i]))
#             if self.antd_start_pos >= start  and self.antd_start_pos <= end:
#                 return i
#             start = end + 1
#         return len(self.sents) - 1
    
#     def get_sent_distance(self) -> int:
#         anph_sent_id = self.get_anph_sent_id()
#         antd_sent_id = self.get_antd_sent_id()
#         return anph_sent_id - antd_sent_id
    
#     def get_start_idx_antd_relative(self) -> int:
#         start = sum([len(str(self.sents[i])) + 1 for i in range(self.antd_sent_id)])
#         return self.antd_start_pos - start
    
#     def relative_pos_start_end_token_antd_sent(self) -> int:
#         doc_antd  = nlp(self.antd)
#         match = True
#         for i, _ in enumerate(self.sents[self.antd_sent_id]):
#             for j, token_antd in enumerate(doc_antd):
#                 if self.sents[self.antd_sent_id][i+j].text != token_antd.text:
#                     match = False
#                     break
#             if match:
#                 return i, i + len(doc_antd) - 1
#             match = True 
#         return 0, len(self.sents[self.antd_sent_id]) - 1
            
#     def get_end_idx_antd_relative(self) -> int:
#         last_idx = self.antd_end_pos
#         while self.data[last_idx - 1] != ' ':
#             last_idx -= 1
#         start = sum([len(str(self.sents[i])) + 1 for i in range(self.antd_sent_id)])
#         return last_idx - start
    
#     def get_verb_token_id(self) -> int:
#         antd_sent_id = self.get_antd_sent_id()
#         antd_start_token_id, antd_end_token_id = self.relative_pos_start_end_token_antd_sent()
#         for i, token in enumerate(self.sents[antd_sent_id]):
#             if token.dep_ == 'ROOT' and i >= antd_start_token_id and i <= antd_end_token_id:
#                 print('got root verb', token.text)
#                 return i
            
#         for i in range(len(self.sents[antd_sent_id]) -1, -1, -1):
#             token = self.sents[antd_sent_id][i]
#             if token.pos_ == 'VERB' and i >= antd_start_token_id and i <= antd_end_token_id:
#                 print('got last verb', token.text)
#                 return i
            
#         for i in range(len(self.sents[antd_sent_id]) -1, -1, -1):
#             token = self.sents[antd_sent_id][i]
#             if token.dep_ == 'ROOT':
#                 print('returning dep', token.text)
#                 return i
#         return 0 

#     def get_log_token_distance(self) -> float:
        
#         if self.antd_sent_id < self.anph_sent_id:
#             token_distance = len(self.sents[self.antd_sent_id]) - self.get_verb_token_id() + self.relative_pos_anph_sent()
#             print('diff sent token distance')
#             return math.log(token_distance)
#         else:
#             print('same sent token distance')
#             anph_pos, verb_pos = self.relative_pos_anph_sent(), self.get_verb_token_id()
#             return math.log(anph_pos - verb_pos) if anph_pos > verb_pos else 0
        
#     def get_relative_positon(self) -> bool:
#         if self.antd_sent_id < self.anph_sent_id:
#             return True
#         else:
#             anph_pos, verb_pos = self.relative_pos_anph_sent(), self.get_verb_token_id()
#             return True if anph_pos > verb_pos else False
        
#     def get_direct_dominance(self) -> bool:
#         if self.antd_sent_id < self.anph_sent_id:
#             return False
#         else:
#             verb_pos = self.get_verb_token_id()
#             anph = self.pronoun_of_anph()
#             for token in self.sents[self.antd_sent_id][verb_pos].children:
#                 if token.text == anph:
#                     return True
#             return False
        
#     def get_dominance(self) -> bool:
#         if self.antd_sent_id < self.anph_sent_id:
#             return False
#         else:
#             verb_pos = self.get_verb_token_id()
#             verb = self.sents[self.antd_sent_id][verb_pos].text
#             if self.sents[self.antd_sent_id][verb_pos].dep_ ==  "ROOT":
#                 return True
#             token_pos = self.relative_pos_anph_sent()
#             token = self.sents[self.antd_sent_id][token_pos]
#             while token.dep_ != "ROOT":
#                 if token.text == verb:
#                     return True
#                 token = token.head
#             return False
        
#     def get_candidate_path(self) -> str:
#         verb_pos = self.get_verb_token_id()
#         token = self.sents[self.antd_sent_id][verb_pos]
#         if token.dep_ == "ROOT":
#             return "ROOT"
#         if token.head.text == token.text:
#             return token.dep_
#         return ''
    
#     def get_negated_candidate(self) -> bool:
#         verb_pos = self.get_verb_token_id()
#         token = self.sents[self.antd_sent_id][verb_pos]        
#         for child in token.children:
#             if child.dep_ == 'neg':
#                 return True 
#         return False
        
#     def get_candidate_transitivity(self): 
#         verb_pos = self.get_verb_token_id()
#         token = self.sents[self.antd_sent_id][verb_pos]  
#         for child in token.children:
#             if "obj" in child.dep_:
#                 return True
#         return False
    

class HelperAntecedent(HelperClassifier):
    def __init__(self, label: dict, datapoint: dict, label_anph_obj: dict, label_antd_obj: dict) -> None:
        super().__init__(label, datapoint, label_anph_obj, label_antd_obj)
        self.antd_start_pos = self.label_antd_obj.get("start")
        self.antd_end_pos = self.label_antd_obj.get("end")
        self.antd = self.data[self.antd_start_pos: self.antd_end_pos+1]
        self.antd_sent_id = self.get_antd_sent_id()
        
    def get_antd_sent_id(self) -> int:
        start = 0
        for i in range(len(self.sents)):
            end = start + len(str(self.sents[i]))
            if self.antd_start_pos >= start  and self.antd_start_pos <= end:
                return i
            start = end + 1
        return len(self.sents) - 1
    
    def relative_pos_start_end_token_antd_sent(self) -> int:
        doc_antd  = nlp(self.antd)
        match = True
        for i, _ in enumerate(self.sents[self.antd_sent_id]):
            for j, token_antd in enumerate(doc_antd):
                try:
                    if self.sents[self.antd_sent_id][i+j].text != token_antd.text:
                        match = False
                        break
                except:
                    continue
            if match:
                return i, i + len(doc_antd) - 1
            match = True 
        return 0, len(self.sents[self.antd_sent_id]) - 1
    
    def get_verb_candidates(self) -> list:
        candidates = list()
        n_prev_sent = 3 
        for i in range(self.anph_sent_id, -1 , -1):
            if n_prev_sent < 0:
                break
            sent = self.sents[i]
            for j, token in enumerate(sent):
                if token.pos_ == 'VERB':
                    candidates.append((i,j))
            n_prev_sent -= 1

        return candidates
    
    def get_sent_distance(self, sent_id) -> int:
        return sent_id - self.anph_sent_id
    
    def get_log_token_distance(self, sent_id, cand_vb_id) -> int:
        if sent_id < self.anph_sent_id:
            token_distance = sum([len(self.sents[i]) for i in range(sent_id + 1, self.anph_sent_id)])
            token_distance += len(self.sents[sent_id]) - cand_vb_id + self.relative_pos_anph_sent()
            return math.log(token_distance)
        else:
            if cand_vb_id < self.relative_pos_anph_sent():
                token_distance = self.relative_pos_anph_sent() - cand_vb_id
                return math.log(token_distance)
            else:
                return 0
            
    def get_relative_positon(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return True
        else:
            return True if cand_vb_id < self.relative_pos_anph_sent() else False
        
    def get_direct_dominance(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return False
        else:
            anph_token = self.sents[self.anph_sent_id][self.relative_pos_anph_sent()]
            verb_token = self.sents[sent_id][cand_vb_id]
            for child in verb_token.children:
                if child.text == anph_token.text:
                    return True
            return False
    
    def get_dominance(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return False
        else:
            anph_token = self.sents[self.anph_sent_id][self.relative_pos_anph_sent()]
            verb_token = self.sents[sent_id][cand_vb_id]
            while anph_token.dep_ != 'ROOT':
                if anph_token.text == verb_token.text:
                    return True
                anph_token = anph_token.head
            return False
        
    def get_candidate_path(self, sent_id, cand_vb_id) -> str:
        verb_token = self.sents[sent_id][cand_vb_id]
        if verb_token.dep_ == 'ROOT':
            return 'ROOT'
        elif verb_token.head.dep_ == 'ROOT':
            return verb_token.dep_
        else:
            return ''
        
    def get_negated_candidate(self, sent_id, cand_vb_id) -> bool:
        verb_token = self.sents[sent_id][cand_vb_id]
        for child in verb_token.children:
            if child.dep_ == 'neg':
                return True
        return False
    
    def get_candidate_transitivity(self, sent_id, cand_vb_id) -> str:
        verb_token = self.sents[sent_id][cand_vb_id]
        for child in verb_token.children:
            if 'obj' in child.dep_:
                return True
        return False

    def is_antd(self, sent_id, cand_vb_id) -> bool:
        antd_start_token_id, antd_end_token_id = self.relative_pos_start_end_token_antd_sent()
        if cand_vb_id >= antd_start_token_id and cand_vb_id <= antd_end_token_id:
            return True
        else:
            return False