from data_utils import extract_features
from paraphrase import perturb_text
import numpy as np
import spacy
import math

import random


nlp = spacy.load('en_core_web_lg')

# Node class
class Node:
    
    def __init__(self, 
                 text, 
                 root=None, 
                 grad_guide=None, 
                 parent=None,
                 candidates_dict=None,
                 chosen_index=None, 
                 indexes_already_used=None, 
                 level=0):
        self.text = text
        self.root = root if root != None else self
        self.grad_guide = grad_guide
        self.parent = parent if parent != None else self
        self.candidates_dict = candidates_dict
        self.chosen_index = chosen_index
        self.indexes_already_used = indexes_already_used if indexes_already_used != None else []
        self.level = level
        #These will be updated in the goal and expand function
        self.features = []
        self.prob = None
        self.cl = None


    
    def expand(self, n_changes_per_level=None, 
               max_depth_level=None, 
               most_salient=True, 
               use_typos=False, use_homoglyphs=False, 
               max_length=1000, verbose=False):

        if max_depth_level == None:
            max_depth_level=7
        
        if self.level > max_depth_level:
            return []
        
        # Compute the Forward Gradient
        model = self.grad_guide.model
        if not len(self.features):
            self.features = extract_features([self.text], max_length=max_length)[0].reshape(1, -1)
        grads = self.grad_guide.wordwise_grads(self.features).squeeze()
        
        indexes_to_use = sorted(np.setdiff1d(range(len(self.text)), self.indexes_already_used), 
                              key=lambda k: grads[k], 
                              reverse=most_salient)

        n_changes = 0
        perturbed_texts = []
        for index in indexes_to_use:
            if index in self.candidates_dict:
                n_changes += 1
                perturbed_texts += perturb_text(self.text, 
                                                index,
                                                self.candidates_dict[index])
                if n_changes == n_changes_per_level:
                    break

        if verbose:
            print("Level: {} Npert: {} Text: {}".format(self.level, len(perturbed_texts), self.text))

        children = np.empty([len(perturbed_texts)], dtype=Node)
        for index, perturbed_text in enumerate(perturbed_texts):

            indexes_already_used=self.indexes_already_used.copy()
            indexes_already_used.append(perturbed_text[1])
            children[index] = Node(nlp(perturbed_text[0]), 
                                 self.root, 
                                 self.grad_guide, 
                                 self, #parent of the child
                                 self.candidates_dict,
                                 perturbed_text[1],
                                 indexes_already_used, 
                                 self.level+1)


        return children
    
    def __repr__(self):
        return '{}'.format(self.text)