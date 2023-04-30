import json
import textacy
import spacy
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
        count = 0
        for i in range(relative_idx, -1, -1):
                if anph_sent[i] == ' ':
                    count += 1

        return count + 1
        
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
    

class HelperAntecedent:
    def __init__(self) -> None:
        pass