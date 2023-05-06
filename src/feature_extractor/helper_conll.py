from collections import defaultdict
from tqdm import tqdm
from typing import Union
import math
import textacy
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans

nlp = spacy.load("en_core_web_sm")

from datasets import load_dataset

def load_conll_data():
    dataset = load_dataset("conll2012_ontonotesv5", "english_v4")
    return dataset

def transform_conll_data():
    dataset = load_conll_data()
    data = defaultdict(lambda: [[],[]])
    for n, item in enumerate(dataset['train']):
        for sent in item['sentences']:
            part_id = sent['part_id']
            part_sent = sent['words']
            coref_spans = sent['coref_spans']
            data[(n, part_id)][0].append(part_sent)
            data[(n, part_id)][1].append(coref_spans)
    return data

def get_filter_conll_data():
    data = transform_conll_data()      
    filtered_data = defaultdict(lambda: [[],[]])
    for key in data:
        for i, sent in enumerate(data[key][0]):
            for j, word in enumerate(sent):
                if word == 'this' or word == 'This' or word == 'that' or word == 'That':
                    for cluster, start, end in data[key][1][i]:
                        if start == end and start == j:
                            filtered_data[key] = data[key]

    return filtered_data

class HelperConllClassifier:
    def __init__(self, sents, anph_sent_id, anph_id) -> None:
        self.sents = sents
        self.anph_sent_id = anph_sent_id
        self.anph_id = anph_id
        self.anph = sents[anph_sent_id][anph_id]
        self.sent = ' '.join(sents[anph_sent_id])
        self.doc = nlp(self.sent)

    def get_verb_presence(self) -> int:
        patterns = [{"POS": "VERB"}]
        count = 0
        for i in range(self.anph_sent_id -1, -1, -1):
            doc_sent = nlp(' '.join(self.sents[i]))
            verbs = textacy.extract.token_matches(doc_sent[:], patterns=patterns)
            if sum(1 for _ in verbs) > 0:
                count += 1
        return count
    
    def get_parent_lemma(self) -> str:
        for token in self.doc:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.text == self.anph:
                        return token.lemma_
                    
        return ''
    
    def get_parent_and_label(self) -> Union[str, str]:
        for token in self.doc:
            if token.text == self.anph and token.has_head:
                return token.head.lemma_,token.dep_
        return '', ''    
    
    def get_negated_parent(self) -> bool:
        for token in self.doc:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.dep_ == 'neg':
                        return True
        return False

    def get_parent_transitivity(self) -> bool:
        for token in self.doc:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if "obj" in child.dep_:
                        return True
        
        return False

    def get_pronoun_path(self) -> str:
        for token in self.doc:
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if str(child) == self.anph:
                        return child.dep_
        return ''
    
    def is_abstract_anaphora(self, phrase) -> str:
        doc_phrase = nlp(' '.join(phrase))
        pattern = [[{'POS': 'VERB', 'OP': '?'},
                    {'POS': 'ADV', 'OP': '*'},
                    {'POS': 'VERB', 'OP': '+'}]]
        matcher = Matcher(nlp.vocab)
        matcher.add("Verb phrase", pattern)
        matches = matcher(doc_phrase)
        return True if len(matches) > 0 else False
    

class HelperConllAntecedent(HelperConllClassifier):
    def __init__(self, sents, anph_sent_id, anph_id) -> None:
        super().__init__(sents, anph_sent_id, anph_id)

    def get_verb_candidates(self):
        candidates = list()
        n_prev_sent = 3 
        for i in range(self.anph_sent_id, -1 , -1):
            if n_prev_sent < 0:
                break
            doc_sent = nlp(' '.join(self.sents[i]))
            for j, token in enumerate(doc_sent):
                if token.pos_ == 'VERB':
                    candidates.append((i,j))
            n_prev_sent -= 1

        return candidates
    
    def get_sent_distance(self, sent_id):
        return sent_id - self.anph_sent_id
    
    def get_log_token_distance(self, sent_id, cand_vb_id) -> int:
        if sent_id < self.anph_sent_id:
            token_distance = sum([len(self.sents[i]) for i in range(sent_id + 1, self.anph_sent_id)])
            token_distance += len(self.sents[sent_id]) - cand_vb_id + self.anph_id
            return math.log(token_distance) if token_distance > 0 else 0
    
        else:
            if cand_vb_id < self.anph_id:
                token_distance = self.anph_id - cand_vb_id
                return math.log(token_distance)
            else:
                return 0

    def get_relative_positon(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return True
        else:
            return True if cand_vb_id < self.anph_id else False
        
    def get_direct_dominance(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return False
        else:
            doc_anph_sent  = nlp(' '.join(self.sents[self.anph_sent_id]))
            anph_token = doc_anph_sent[self.anph_id]
            verb_token = doc_anph_sent[cand_vb_id]
            for child in verb_token.children:
                if child.text == anph_token.text:
                    return True
            return False
        
    def get_dominance(self, sent_id, cand_vb_id) -> bool:
        if sent_id < self.anph_sent_id:
            return False
        else:
            doc_anph_sent  = nlp(' '.join(self.sents[self.anph_sent_id]))
            anph_token = doc_anph_sent[self.anph_id]
            verb_token = doc_anph_sent[cand_vb_id]
            while anph_token.dep_ != 'ROOT':
                if anph_token.text == verb_token.text:
                    return True
                anph_token = anph_token.head
            return False
        
    def get_candidate_path(self, sent_id, cand_vb_id) -> str:
        doc_sent = nlp(' '.join(self.sents[sent_id]))
        verb_token = doc_sent[cand_vb_id]
        if verb_token.dep_ == 'ROOT':
            return 'ROOT'
        elif verb_token.head.dep_ == 'ROOT':
            return verb_token.dep_
        else:
            return ''
        
    def get_negated_candidate(self, sent_id, cand_vb_id) -> bool:
        doc_sent = nlp(' '.join(self.sents[sent_id]))
        verb_token = doc_sent[cand_vb_id]
        for child in verb_token.children:
            if child.dep_ == 'neg':
                return True
        return False
    
    def get_candidate_transitivity(self, sent_id, cand_vb_id) -> str:
        doc_sent = nlp(' '.join(self.sents[sent_id]))
        verb_token = doc_sent[cand_vb_id]
        for child in verb_token.children:
            if 'obj' in child.dep_:
                return True
        return False
    
    def is_antd(self, sent_id, cand_vb_id, anph_sent_id, prev_start, prev_end) -> bool:
        if sent_id == anph_sent_id and cand_vb_id >= prev_start and cand_vb_id <= prev_end:
            return True
        else:
            return False


