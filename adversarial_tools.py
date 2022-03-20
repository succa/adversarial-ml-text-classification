import keras
import spacy
import numpy as np
import tensorflow as tf
import time

from keras import backend as K

from data_utils import extract_features
from paraphrase import perturb_text, generate_candidates
from graph_search import *
from node import Node

from timeout import timeout


nlp = spacy.load('en_core_web_lg', tagger=False, entity=False)


class ForwardGradWrapper:
    '''
    Utility class that computes the gradient of model probability output
    with respect to model input.
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''

        input_tensor = model.input
        embedding_tensor = model.layers[0](input_tensor)
        output_tensor = embedding_tensor
        for layer in model.layers[1:-2]:
            output_tensor = layer(output_tensor)
        dense_input = model.layers[-2](output_tensor)
        output_tensor = model.layers[-1](dense_input)
        grad_tensor, = tf.gradients(output_tensor, embedding_tensor)
        grad_sum_tensor = tf.reduce_sum(grad_tensor, reduction_indices=2)

        self.model = model
        self.grad_tensor = grad_tensor
        self.dense_input = dense_input
        self.input_tensor = input_tensor
        self.grad_sum_tensor = grad_sum_tensor

    def wordwise_grads(self, feature_vectors):
        sess = K.get_session()
        grad_sum = sess.run(self.grad_sum_tensor, feed_dict={
            self.input_tensor: feature_vectors,
            keras.backend.learning_phase(): 0
        })
        return grad_sum

    def get_dense_input(self, feature_vectors):
        sess = K.get_session()
        grad_sum = sess.run(self.dense_input, feed_dict={
            self.input_tensor: feature_vectors,
            keras.backend.learning_phase(): 0
        })
        return grad_sum

###########################
def _cost_fn_base(node, max_length=1000):
    return 0

def _heuristic_fn_base(node, epsilon, max_length=1000):
    return 0
############################
        
def _goal_fn(node, max_length=1000):
    """Tell whether the example has reached the goal."""
    if not len(node.features):
        node.features = extract_features([node.text], max_length=max_length)[0].reshape(1, -1)
    if node.cl==None:
        node.cl = node.grad_guide.model.predict_classes(node.features, verbose=0).squeeze()
        
    if not len(node.root.features):
        node.root.features = extract_features([node.root.text], max_length=max_length)[0].reshape(1, -1)
    if node.root.cl==None:
        node.root.cl = node.root.grad_guide.model.predict_classes(node.root.features, verbose=0).squeeze()
    return node.cl != node.root.cl

def _expand_fn(node, max_depth_level=None, cost_fn=None, n_changes_per_level=None, most_salient=True, 
               use_typos=False, use_homoglyphs=False, max_length=1000, verbose=False):
    if cost_fn==None:
        cost_fn=_cost_fn_base
    children = node.expand(n_changes_per_level=n_changes_per_level, 
                           max_depth_level=max_depth_level, 
                           most_salient=most_salient, 
                           use_typos=use_typos, use_homoglyphs=use_homoglyphs, 
                           max_length=max_length, verbose=verbose)
    costs = [cost_fn(child, max_length=max_length) for child in children]
    return list(zip(children, costs))

# Set the timeout here
@timeout(1800)
def find_adversarial(doc, grad_guide, search_algorithm='a_star', return_path=False, iter_lim=None,
                     max_depth_level=None,
                     cost_fn=None, heuristic_fn=None, 
                     n_changes_per_level=None, most_salient=True,
                     use_synonyms=True,
                     use_typos=False, 
                     use_homoglyphs=False,
                     epsilon=1, max_length=1000, verbose=False):
#     if cost_fn==None:
#         cost_fn = _cost_fn_base
#     if heuristic_fn==None:
#         heuristic_fn = _heuristic_fn_base
        
    if search_algorithm not in ['a_star', 'ida_star', 'hill_climbing', 'bfs']:
        raise ValueError("Unknown search algorithm")
    
    #First precompute all the possible candidates for each word in doc
    candidates_dict = generate_candidates(doc, use_synonyms=use_synonyms, use_typos=use_typos, use_homoglyphs=use_homoglyphs)
    #Generate the Root node
    root = Node(doc, grad_guide=grad_guide, candidates_dict=candidates_dict)
    
    if search_algorithm == 'a_star':
        time_start = time.time()
        (node, cost) = a_star_search(
            start_node=root, 
            expand_fn=lambda x: _expand_fn(x, max_depth_level, cost_fn, n_changes_per_level, most_salient, 
                                           use_typos, use_homoglyphs, max_length=max_length, verbose=verbose), 
            goal_fn=lambda x: _goal_fn(x, max_length=max_length), 
            heuristic_fn=lambda x: heuristic_fn(x, epsilon=epsilon, max_length=max_length),
            iter_lim=iter_lim,
            return_path=return_path
        )
        time_stop = time.time()
    elif search_algorithm == 'ida_star':
        time_start = time.time()
        (node, cost) = ida_star_search(
            start_node=root, 
            expand_fn=lambda x: _expand_fn(x, max_depth_level, cost_fn, n_changes_per_level, most_salient, 
                                           use_typos, use_homoglyphs, max_length=max_length, verbose=verbose), 
            goal_fn=lambda x: _goal_fn(x, max_length=max_length), 
            heuristic_fn=lambda x: heuristic_fn(x, epsilon=epsilon, max_length=max_length),
            iter_lim=iter_lim,
            return_path=return_path
        )
        time_stop = time.time()
    elif search_algorithm == 'hill_climbing':
        time_start = time.time()
        (node, cost) = hill_climbing_search(
            start_node=root, 
            expand_fn=lambda x: _expand_fn(x, max_depth_level, cost_fn, n_changes_per_level, most_salient, 
                                           use_typos, use_homoglyphs, max_length=max_length, verbose=verbose), 
            goal_fn=lambda x: _goal_fn(x, max_length=max_length), 
            heuristic_fn=lambda x: heuristic_fn(x, epsilon=epsilon, max_length=max_length),
            iter_lim=iter_lim,
            return_path=return_path
        )
        time_stop = time.time()
    elif search_algorithm == 'bfs':
        time_start = time.time()
        (node, cost) = a_star_search(
            start_node=root, 
            expand_fn=lambda x: _expand_fn(x, max_depth_level, _cost_fn_base, n_changes_per_level, most_salient, 
                                           use_typos, use_homoglyphs, max_length=max_length, verbose=verbose), 
            goal_fn=lambda x: _goal_fn(x, max_length=max_length), 
            heuristic_fn=lambda x: heuristic_fn(x, epsilon=epsilon, max_length=max_length),
            iter_lim=iter_lim,
            return_path=return_path
        )
        time_stop = time.time()
        

    

    X = extract_features([doc], max_length=max_length).reshape(1,-1)
    confidence = grad_guide.model.predict_proba(X, verbose=0).squeeze().tolist()
    
    adv_pred = None
    adv_confidence = None
    if node != None:
        X_adv = extract_features([node.text], max_length=max_length).reshape(1,-1)
        adv_pred = grad_guide.model.predict_classes(X_adv, verbose=0).squeeze().tolist()
        adv_confidence = grad_guide.model.predict_proba(X_adv, verbose=0).squeeze().tolist()

    return (node, 
            adv_pred, 
            confidence,
            adv_confidence, 
            cost, 
           (time_stop-time_start))


