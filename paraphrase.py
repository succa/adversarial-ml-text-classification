import attr
import nltk
import spacy

from collections import OrderedDict
from functools import partial

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk as disambiguate

from typos import typos
from homoglyphs import homoglyphs

nlp = spacy.load('en_core_web_lg')


# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
supported_pos_tags = [
    # 'CC',   # coordinating conjunction
    # 'CD',   # Cardinal number
    # 'DT',   # Determiner
    # 'EX',   # Existential there
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction
    'JJ',   # Adjective
    # 'JJR',  # Adjective, comparative
    # 'JJS',  # Adjective, superlative
    # 'LS',   # List item marker
    # 'MD',   # Modal
    'NN',   # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS', # Proper noun, plural
    # 'PDT',  # Predeterminer
    # 'POS',  # Possessive ending
    # 'PRP',  # Personal pronoun
    # 'PRP$', # Possessive pronoun
    'RB',   # Adverb
    # 'RBR',  # Adverb, comparative
    # 'RBS',  # Adverb, superlative
    # 'RP',   # Particle
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection
    'VB',   # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner
    # 'WP',   # Wh-pronoun
    # 'WP$',  # Possessive wh-pronoun
    # 'WRB',  # Wh-adverb
]

def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    try:
        pos = spacy_token.tag_[0].lower()
        if pos in ['a', 'n', 'v']:
            return pos
    except:
        return None

def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    #Attempt to handle TypeError Bug
    try:
        similarity = doc[start: original.i + window_size].similarity(synonym)
    except TypeError:
        similarity = 0
    return similarity

def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if  (len(synonym.text.split()) > 2) or \
        (synonym.lemma == token.lemma) or \
        (synonym.tag != token.tag) or \
        (token.text.lower() == 'be'):
        return False
    else:
        return True

def _generate_synonym_candidates(doc, index, disambiguate=False, rank_fn=None):

    if rank_fn is None:
        rank_fn=vsm_similarity
        
    token = doc[index]
    wordnet_pos = _get_wordnet_pos(token)
    wordnet_synonyms = []
    if disambiguate:
        try:
            synset = disambiguate(doc.text, token.text, pos=wordnet_pos)
            wordnet_synonyms = synset.lemmas()
        except:
            return
    else:
        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

    synonyms = []
    for wordnet_synonym in wordnet_synonyms:
        spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
        synonyms.append(spacy_synonym)

    synonyms = filter(partial(_synonym_prefilter_fn, token),
                        synonyms)
    synonyms = reversed(sorted(synonyms,
                         key=partial(rank_fn, doc, token)))
    synonyms = map(lambda x: x.text, synonyms)
    
    return list(synonyms)
    
def _generate_typo_candidates(doc, index, min_token_length=4, rank=1000):
    token = doc[index]
    if (len(token)) < min_token_length:
        return []
    return list(typos(token.text))

def _generate_homoglyph_candidates(doc, index, min_token_length=4, rank=1000):
    token = doc[index]
    if (len(token)) < min_token_length:
        return []
    return list(homoglyphs(token.text))
        
def perturb_text(doc, index, condidates):
    
    perturbed_texts = []
    for candidate in condidates:
        perturbed_text = []
        for position, token in enumerate(doc):
            if position == index:
                perturbed_text.append(candidate)
            else:
                perturbed_text.append(token.text)
        perturbed_texts.append((' '.join(perturbed_text), index))
    
    return perturbed_texts

def filter_not_in_vocabulary(candidates):
    boolean = False
    for i in reversed(range(0, len(candidates))):
        if candidates[i] not in nlp.vocab and boolean:
            del candidates[i]
        boolean=True
    return candidates

def generate_candidates(doc, use_synonyms=True, use_typos=False, use_homoglyphs=False, rank_fn=None, filter_not_in_voc=True):
    dictionary = {}
    for index in range(0,len(doc)):
        candidates = []
        if use_synonyms:
            candidates.extend(_generate_synonym_candidates(doc, index, rank_fn=rank_fn))
        if use_typos:
            candidates.extend(_generate_typo_candidates(doc, index))
        if use_homoglyphs:
            candidates.extend(_generate_homoglyph_candidates(doc, index))
           
        if filter_not_in_voc:
            candidates = filter_not_in_vocabulary(candidates)
        
        if candidates != []:
            dictionary[index] = candidates
            
    return dictionary

