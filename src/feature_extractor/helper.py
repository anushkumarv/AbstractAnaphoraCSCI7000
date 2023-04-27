import textacy
import spacy

from labels import get_lables

nlp = spacy.load("en_core_web_sm")

label_non_abs_anph_set, label_non_abs_antd_set, label_abs_anph_set, label_abs_antd_set = get_lables()

class Helper:
    def __init__(self, label: dict, datapoint: dict, label_anph_obj: dict, label_antd_obj: dict) -> None:
        self.label = label
        self.datapoint = datapoint
        self.label_anph_obj = label_anph_obj
        self.label_antd_obj = label_antd_obj
        self.data = self.datapoint.get("data").get("corefText")
        self.anph_start_pos = self.label_anph_obj.get("start")
        self.anph_end_pos = self.label_anph_obj.get("end")
        self.doc = nlp(datapoint.get("data").get("corefText"))

    def pronoun_of_anph(self) -> str:
        return self.label.get("value").get("text")

    def anph_sent_id(self) -> int:
        sents = list(self.doc.sents)
        start = 0
        for i in range(len(sents)):
            end = start + len(str(sents[i]))
            if self.anph_start_pos >= start  and self.anph_start_pos <= end:
                return i
            start = end + 1
        return len(sents) - 1


    def start_idx_anph_relative(self) -> int:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        start = sum([len(str(sents[i])) + 1 for i in range(anph_sent_id)])
        return self.anph_start_pos - start


    def relative_pos_anph_sent(self) -> int:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        anph_sent = str(sents[anph_sent_id])
        relative_idx = self.start_idx_anph_relative()
        count = 0
        for i in range(relative_idx, -1, -1):
                if anph_sent[i] == ' ':
                    count += 1

        return count + 1

        
    def get_verb_presence(self) -> int:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        patterns = [{"POS": "VERB"}]
        count = 0
        for i in range(anph_sent_id -1, -1, -1):
            verbs = textacy.extract.token_matches(sents[i], patterns=patterns)
            if sum(1 for _ in verbs) > 0:
                count += 1
        return count
    
    def get_pronoun_path(self) -> str:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        anph = self.pronoun_of_anph()
        for token in sents[anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if str(child) == anph:
                        return child.dep_
        return ''
    
    def get_parent_lemma(self) -> str:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        anph = self.pronoun_of_anph()
        for token in sents[anph_sent_id]:
            if token.dep_ == 'ROOT':
                return token.lemma_
            
        return ''
    
    def get_negated_parent(self) -> bool:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        anph = self.pronoun_of_anph()
        for token in sents[anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.dep_ == 'neg':
                        return True
        return False
    
    def get_parent_transitivity(self) -> bool:
        sents = list(self.doc.sents)
        anph_sent_id = self.anph_sent_id()
        anph = self.pronoun_of_anph()
        for token in sents[anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if "obj" in child.dep_:
                        return True
        
        return False
    
    def is_abstract_anaphora(self) -> bool:
        return True if self.label_anph_obj.get("labels")[0] in label_abs_anph_set else False
    