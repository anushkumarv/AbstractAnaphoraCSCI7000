from collections import defaultdict
from tqdm import tqdm
from typing import Union
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
    

        

