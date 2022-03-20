import csv
import argparse

from datetime import datetime

import spacy

import model as model_config
from adversarial_tools import ForwardGradWrapper, find_adversarial
from data_utils import load as load_data, extract_features
from graph_search import *
from paraphrase import *
from node import Node
import numpy as np
import scipy
import sys


parser = argparse.ArgumentParser(
        description='Craft adversarial examples for a text classifier.')
parser.add_argument('--model_path',
                    help='Path to model weights',
                    default='./data/model.dat')
parser.add_argument('--adversarial_texts_path',
                    help='Path where results will be saved',
                    default='./data/adversarial_texts.csv')
parser.add_argument('--test_samples_cap',
                    help='Amount of test samples to use',
                    type=int, default=100)
parser.add_argument('--search_algorithm',
                    help='Search Algorithm [a_star, ida_star, hill_climbing, bfs]',
                    default='a_star')
parser.add_argument('--heuristic',
                    help='Heuristic Function [None, difference, distance]',
                    default='distance')
parser.add_argument('--cost',
                    help='Cost Function [None, semantic, L2norm]',
                    default='L2norm')
parser.add_argument('--most_salient',
                    help='change the most salient words',
                    action='store_true')
parser.add_argument('--expansion',
                    help='number of changes per level',
                    type=int, default=None)
parser.add_argument('--use_synonyms',
                    help='Whether to use synonyms for paraphrases',
                    action='store_true')
parser.add_argument('--use_typos',
                    help='Whether to use typos for paraphrases',
                    action='store_true')
parser.add_argument('--use_homoglyphs',
                    help='Whether to use homoglyphs for paraphrases',
                    action='store_true')
parser.add_argument('--epsilon',
                    help='epsilon',
                    type=int, default=1)
parser.add_argument('--max_depth_level',
                    help='max_depth_level',
                    type=int, default=3)

# Cost Functions
def cost_fn_none(node, max_length=1000):
    return 0

def cost_fn_semantic(node, max_length=1000):
    window_size = 3
    start = max(0, node.chosen_index - window_size)
    try:
        similarity = node.root.text[start: node.chosen_index + window_size].similarity(node.text[node.chosen_index])
    except TypeError:
        similarity = 0
    return 1 - similarity

def cost_fn_L2norm(node, max_length=1000):
    if not len(node.parent.features):
        node.parent.features = extract_features([node.parent.text], max_length=max_length)[0].reshape(1, -1)
    if not len(node.features):
        node.features = extract_features([node.text], max_length=max_length)[0].reshape(1, -1)

    parent_before_dense = node.grad_guide.get_dense_input(node.parent.features)[0]
    node_before_dense = node.grad_guide.get_dense_input(node.features)[0]
    return np.linalg.norm(parent_before_dense - node_before_dense, ord=2)

def cost_fn(cost_function):
    cost_fn_map = {
        'None': cost_fn_none,
        'semantic': cost_fn_semantic,
        'L2norm': cost_fn_L2norm
    }
    return cost_fn_map[cost_function]

# Heuristic Fucntions
def heuristic_fn_none(node, epsilon=1, q_norm=2, max_length=1000):
    return 0

def heuristic_fn_difference(node, epsilon=1, q_norm=2, max_length=1000):
    if not len(node.features):
        node.features = extract_features([node.text], max_length=max_length)[0].reshape(1, -1)
    if node.prob==None:
        node.prob = node.grad_guide.model.predict_proba([node.features], verbose=0).squeeze()

    if not len(node.root.features):
        node.root.features = extract_features([node.root.text], max_length=max_length)[0].reshape(1, -1)
    if node.root.prob==None:
        node.root.prob = node.root.grad_guide.model.predict_proba([node.root.features], verbose=0).squeeze()
    if node.root.cl==None:
        node.root.cl = node.root.grad_guide.model.predict_classes(node.root.features, verbose=0).squeeze()    

    if node.root.cl == 1:
        return node.prob - node.root.prob
    else:
        return node.root.prob - node.prob

def heuristic_fn_distance(node, epsilon=1, q_norm=2, max_length=1000):
    model = node.grad_guide.model

    if not len(node.root.features):
        node.root.features = extract_features([node.root.text], max_length=max_length)[0].reshape(1, -1)
    if node.root.cl==None:
        node.root.cl = node.root.grad_guide.model.predict_classes(node.root.features, verbose=0).squeeze()

    if not len(node.features):
        node.features = extract_features([node.text], max_length=max_length)[0].reshape(1, -1)
    if node.cl==None:
        node.cl = node.grad_guide.model.predict_classes(node.features, verbose=0).squeeze()
    if node.prob==None:
        node.prob = node.grad_guide.model.predict_proba([node.features], verbose=0).squeeze()

    # if same place return 0
    if (node.root.cl == 1 and node.cl == 0) or (node.root.cl == 0 and node.cl == 1) :
        return 0.0
    value = scipy.special.logit(node.prob)
    if abs(value) == np.inf:
        value = 1000000 #Max Value
    h = abs(value) / np.linalg.norm(model.layers[-1].get_weights()[0].squeeze(), ord=q_norm)    
    return h * epsilon

def heuristic_fn(heuristic_function):
    heuristic_fn_map = {
        'None': heuristic_fn_none,
        'difference': heuristic_fn_difference,
        'distance': heuristic_fn_distance
    }
    return heuristic_fn_map[heuristic_function]


if __name__ == '__main__':
    args = parser.parse_args()
    test_samples_cap = args.test_samples_cap
    
    print("Sarch Algorithm: %s" % args.search_algorithm)
    print("Heuristic Function: %s" % args.heuristic)
    print("Cost Function: %s" % args.cost)
    print("Most salient: %s" % args.most_salient)
    print("Expansion: %s" % args.expansion)
    print("Use synonyms: %s" % args.use_synonyms)
    print("Use typos: %s" % args.use_typos)
    print("Use homoglyphs: %s" % args.use_homoglyphs)
    print("Epsilon: %s" % args.epsilon)
    
    if (args.search_algorithm not in ['a_star', 'ida_star', 'hill_climbing', 'bfs']):
        print("Search alrogithm must be [a_star, ida_star, hill_climbing, bfs]")
        sys.exit()
    if (args.cost not in ['None', 'semantic', 'L2norm']):
        print("Cost must be [None, semantic, L2norm]")
        sys.exit()
    if (args.heuristic not in ['None', 'difference', 'distance']):
        print("Cost must be [None, difference, distance]")
        sys.exit()
    
    # Prepare nlp
    nlp = spacy.load('en_core_web_lg')

    # Load Twitter gender data
    (_, _, X_test, y_test), (docs_train, docs_test, _) = \
            load_data('twitter_gender_data', from_cache=True)

    # Load model from weights
    model = model_config.build_model_twitter()
    model.load_weights(args.model_path)

    # Initialize the class that computes forward derivatives
    grad_guide = ForwardGradWrapper(model)

    # Calculate accuracy on test examples
    preds = model.predict_classes(X_test[:test_samples_cap, ]).squeeze()
    accuracy = np.mean(preds == y_test[:test_samples_cap])
    print('Model accuracy on test:', accuracy)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    adversarial_text_data = []
    adversarial_preds = np.array(preds)
    
#     cost_fn = cost_fn(args.cost)
#     heuristic_fn = heuristic_fn(args.heuristic)

    with open(args.adversarial_texts_path, 'w') as handle:
        field = ['index','doc','adv','success','confidence','adv_confidence','cost','changes','indexes','time']
        writer = csv.DictWriter(handle, fieldnames=field)
        writer.writeheader()

        for index, doc in enumerate(docs_test[:test_samples_cap]):
            if (y_test[index] == 0 and preds[index] == 0) or (y_test[index] == 1 and preds[index] == 1):
                print(index)
                try:
                    (node, 
                     adv_pred, 
                     confidence, 
                     adv_confidence, 
                     cost, time) = find_adversarial(doc, 
                                                    grad_guide, 
                                                    search_algorithm=args.search_algorithm,
                                                    return_path=False, 
                                                    max_depth_level=args.max_depth_level,
                                                    cost_fn=cost_fn(args.cost), 
                                                    heuristic_fn=heuristic_fn(args.heuristic),
                                                    n_changes_per_level=args.expansion,
                                                    most_salient=args.most_salient,
                                                    use_synonyms=args.use_synonyms,
                                                    use_typos=args.use_typos, 
                                                    use_homoglyphs=args.use_homoglyphs, 
                                                    epsilon=args.epsilon, verbose=False)
                except Exception as e:
                    print(e)
                    (node, 
                     adv_pred, 
                     confidence, 
                     adv_confidence, 
                     cost, time) = (None, None, None, None, None, 600)

                if node != None:
                    successful_perturbations += 1
                    print('{}. {} ==> {}'.format(index, doc, node.text))
                else:
                    failed_perturbations += 1
                    print('{}. Failure.'.format(index))

                item = ({
                            'index': index,
                            'doc': doc,
                            'adv': node.text if node != None else None,
                            'success': adv_pred != preds[index] if adv_pred != None else False,
                            'confidence': confidence,
                            'adv_confidence': adv_confidence,
                            'cost': cost,
                            'changes': node.level if node != None else None,
                            'indexes': node.indexes_already_used if node != None else None,
                            'time': time
                    })

                writer.writerow(item)
        

    print('Model accuracy on adversarial examples:',
            np.mean(adversarial_preds == y_test[:test_samples_cap]))
    print('Fooling success rate:',
            successful_perturbations / (successful_perturbations + failed_perturbations))
