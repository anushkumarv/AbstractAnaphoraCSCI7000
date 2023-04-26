import textacy
import spacy

nlp = spacy.load("en_core_web_sm")


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

    def sents_before_anph(self) -> int:
        count = 0
        left = self.anph_start_pos
        while left != -1:
            if self.data[left] == '.':
                count += 1
            left -= 1
        return count
        
    def sents_after_anph(self) -> int:
        count = 0
        right = self.anph_start_pos
        while right != len(self.data):
            if self.data[right] == '.':
                count += 1
            right += 1
        return count - 1

    def start_idx_anph_relative(self) -> int:
        left = self.anph_start_pos
        while left != -1 and self.data[left] != '.':
            left -= 1
        if left == -1:
            return self.anph_start_pos
        elif self.data[left] == '.' and self.data[left+1] == ' ':
            return self.anph_start_pos - left - 2
        else:
            return self.anph_start_pos - left - 1


    def relative_pos_anph_sent(self) -> int:
        sents = list(self.doc.sents)
        anph_sent_id = self.sents_before_anph()
        anph_sent = str(sents[anph_sent_id])
        relative_idx = self.start_idx_anph_relative()
        count = 0
        for i in range(relative_idx, -1, -1):
            if anph_sent[i] == ' ':
                count += 1

        return count + 1

        
    def get_verb_presence(self) -> int:
        sents = list(self.doc.sents)
        anph_sent_id = self.sents_before_anph()
        patterns = [{"POS": "VERB"}]
        count = 0
        for i in range(anph_sent_id -1, -1, -1):
            verbs = textacy.extract.token_matches(sents[i], patterns=patterns)
            if sum(1 for _ in verbs) > 0:
                count += 1
        return count
    
    def get_pronoun_path(self) -> str:
        sents = list(self.doc.sents)
        anph_sent_id = self.sents_before_anph()
        anph = self.pronoun_of_anph()
        for token in sents[anph_sent_id]:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if str(child) == anph:
                        return child.dep_
        return ''





        
