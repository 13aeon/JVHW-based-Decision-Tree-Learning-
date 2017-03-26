from __future__ import division
import math
import operator
import time
import copy
import sys
import ast
import csv
from collections import Counter
import numpy as np
import scipy.io as sio
from math import log
from scipy.sparse import csr_matrix
import matlab.engine
##################################################
# data class to hold csv data
##################################################
global jvormle # 0 for JVHW, 1 for MLE
jvormle = 0
global idorc # 0 for ID3, 1 for C4.5
idorc = 0
global eng
poly_entro = None
global nodeoftree
nodeoftree = 0
class data():
    def __init__(self, classifier):
        self.examples = []
        self.attributes = []
        self.attr_types = []
        self.classifier = classifier
        self.class_index = None

##################################################
# function to read in data from the .csv files
##################################################
def read_data(dataset, datafile, datatypes):
    print ("Reading data...")
    f = open(datafile)
    original_file = f.read()
    rowsplit_data = original_file.splitlines()
    dataset.examples = [rows.split(',') for rows in rowsplit_data]

    #list attributes
    dataset.attributes = dataset.examples.pop(0)

    
    #create array that indicates whether each attribute is a numerical value or not
    attr_type = open(datatypes) 
    orig_file = attr_type.read()
    dataset.attr_types = orig_file.split(',')

##################################################
# Preprocess dataset
##################################################
def preprocess2(dataset, existing_class_dict):
    print ("Preprocessing data...")
    if existing_class_dict is None:
        class_values = [example[dataset.class_index] for example in dataset.examples]
        subset = np.sort(np.unique(class_values))
        class_dict = {}
        for i in range(len(subset)):
            class_dict[subset[i]] = str(i)
    else:
        class_dict = existing_class_dict

    for example in dataset.examples:
        if example[dataset.class_index] not in class_dict:
            class_dict[example[dataset.class_index]] = str(len(class_dict))
        example[dataset.class_index] = class_dict[example[dataset.class_index]]

    class_values = [example[dataset.class_index] for example in dataset.examples]
    class_mode = Counter(class_values)
    class_mode = class_mode.most_common(1)[0]
    class_mode = class_mode[0]

    for attr_index in range(len(dataset.attributes)):

        for i in range(len(class_dict)):
            ex_class = filter(lambda x: x[dataset.class_index] == str(i), dataset.examples)
            values_class = [example[attr_index] for example in ex_class]

            if len(values_class):

                values = Counter(values_class)
                value_counts = values.most_common()

                mode = values.most_common(1)[0]
                mode = mode[0]
                if mode == '?':
                    if len(values.most_common(2)) == 1:
                        mode = 0
                    else:
                        mode = values.most_common(2)[1]
                        mode = mode[0]

                attr_modes = [0]*len(dataset.attributes)
                for example in dataset.examples:
                    if (example[attr_index] == '?'):
                        if (example[dataset.class_index] == str(i)):
                            example[attr_index] = mode

        #convert attributes that are numeric to floats
        for example in dataset.examples:
            for x in range(len(dataset.examples[0])):
                if dataset.attributes[x] == 'True':
                    example[x] = float(example[x])
    return class_dict

##################################################
# tree node class that will make up the tree
##################################################
class treeNode():
    def __init__(self, is_leaf, classification, attr_split_index, attr_split_value, parent, upper_child, lower_child, height):
        self.is_leaf = True
        self.classification = None
        self.attr_split = None
        self.attr_split_index = None
        self.attr_split_value = None
        self.parent = parent
        self.upper_child = None
        self.lower_child = None
        self.height = None

##################################################
# compute tree recursively
##################################################

# initialize Tree
    # if dataset is pure (all one result) or there is other stopping criteria then stop
    # for all attributes a in dataset
        # compute information-theoretic criteria if we split on a
    # abest = best attribute according to above
    # tree = create a decision node that tests abest in the root
    # dv (v=1,2,3,...) = induced sub-datasets from D based on abest
    # for all dv
        # tree = compute_tree(dv)
        # attach tree to the corresponding branch of Tree
    # return tree 

def compute_tree(dataset, parent_node, classifier):
    node = treeNode(True, None, None, None, parent_node, None, None, 0)
    if (parent_node == None):
        node.height = 0
    else:
        node.height = node.parent.height + 1
    st_sig = 0; end_sig = 0
    for example in dataset.examples:
        if st_sig == 0:
            comp_eg = example[dataset.class_index]
            st_sig = 1
        else:
            if example[dataset.class_index] != comp_eg:
                end_sig = 1
                break
    if end_sig == 0:
        node.classification = int(example[dataset.class_index])
        node.isleaf = True
        return node
    else:
        node.is_leaf = False

    attr_to_split = None # The index of the attribute we will split on
    max_gain = 0 # The gain given by the best attribute
    split_val = None 
    min_gain = 0.005
    dataset_entropy = calc_dataset_entropy(dataset, classifier)
    for attr_index in range(len(dataset.examples[0])):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None
            attr_value_list = [example[attr_index] for example in dataset.examples] # these are the values we can split on, now we must find the best one
            attr_value_list = list(set(attr_value_list)) # remove duplicates from list of all attribute values
            if(len(attr_value_list) > 100):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total/10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x*ten_percentile])
                attr_value_list = new_list
            attr_value_list = np.sort(attr_value_list)
            for val in attr_value_list:
                # calculate the gain if we split on this value
                # if gain is greater than local_max_gain, save this gain and this value
                local_gain = calc_gain(dataset, dataset_entropy, val, attr_index, classifier) # calculate the gain if we split on this value
  
                if (local_gain > local_max_gain):
                    local_max_gain = local_gain
                    local_split_val = val

            if (local_max_gain > max_gain):
                max_gain = local_max_gain
                split_val = local_split_val
                attr_to_split = attr_index

    #attr_to_split is now the best attribute according to our gain metric
    #if (split_val is None or attr_to_split is None):
    #    print ("Something went wrong. Couldn't find an attribute to split on or a split value.")
    if ( max_gain <= min_gain or node.height > 20 or split_val is None or attr_to_split is None):#max_gain <= min_gain or

        node.is_leaf = True
        node.classification = classify_leaf(dataset, classifier)

        return node

    node.attr_split_index = attr_to_split
    node.attr_split = dataset.attributes[attr_to_split]
    node.attr_split_value = split_val
    node.attr_split_type = dataset.attr_types[attr_to_split]
    # currently doing one split per node so only two datasets are created
    upper_dataset = data(classifier)
    lower_dataset = data(classifier)
    upper_dataset.attributes = dataset.attributes
    lower_dataset.attributes = dataset.attributes
    upper_dataset.class_index = dataset.class_index
    lower_dataset.class_index = dataset.class_index
    upper_dataset.attr_types = dataset.attr_types
    lower_dataset.attr_types = dataset.attr_types
    if (attr_to_split is not None) and (dataset.attr_types[attr_to_split] == True):
        for example in dataset.examples:
            if (example[attr_to_split] >= split_val):
                upper_dataset.examples.append(example)
            else:
                lower_dataset.examples.append(example)
    else:
        for example in dataset.examples:
            if (example[attr_to_split] == split_val):
                upper_dataset.examples.append(example)
            else:
                lower_dataset.examples.append(example)        

    node.upper_child = compute_tree(upper_dataset, node, classifier)
    node.lower_child = compute_tree(lower_dataset, node, classifier)

    return node

##################################################
# Classify dataset
##################################################
def classify_leaf(dataset, classifier):
    class_index = None
    #find index of classifier
    for a in range(len(dataset.attributes)):
        if dataset.attributes[a] == classifier:
            class_index = a
            break
        else:
            class_index = len(dataset.attributes) - 1
    node_counter = Counter(np.array(dataset.examples)[:,class_index])
    node_counter = node_counter.most_common(1)[0]
    return int(node_counter[0])


##################################################
# Calculate the entropy of the current dataset
##################################################
def calc_dataset_entropy(dataset, classifier):
    global jvormle
    global idorc
    global eng
    class_index = None
    #find index of classifier
    for a in range(len(dataset.attributes)):
        if dataset.attributes[a] == classifier:
            class_index = a
            #print(classifier,dataset.attributes[a],a)
            break
        else:
            class_index = len(dataset.attributes) - 1
    #print(classifier, dataset.attributes)
    if idorc == 2:
        if jvormle ==1:
            entro = eng.est_gini_MLE([float(elements) for elements in list(np.array(dataset.examples)[:,class_index])],2.0)
        else:
            entro = eng.est_gini_JVHW([float(elements) for elements in list(np.array(dataset.examples)[:,class_index])],2.0)
    else:
        if jvormle == 1:
            entro =  est_entro_MLE([float(elements) for elements in list(np.array(dataset.examples)[:,class_index])])
        else:
            entro = est_entro_JVHW([float(elements) for elements in list(np.array(dataset.examples)[:,class_index])])
        entro = entro[0]
    return entro

##################################################
# Calculate the gain of a particular attribute split
##################################################
def calc_gain(dataset, entropy, val, attr_index, classifier):
    classifier2 = dataset.attributes[attr_index]
    attr_entropy = 0
    total_examples = len(dataset.examples);
    gain_upper_dataset = data(classifier)
    gain_lower_dataset = data(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attr_types = dataset.attr_types
    gain_lower_dataset.attr_types = dataset.attr_types
    if dataset.attr_types[attr_index] == True:
        for example in dataset.examples:
            if (example[attr_index] >= val):
                gain_upper_dataset.examples.append(example)
            elif (example[attr_index] < val):
                gain_lower_dataset.examples.append(example)
    else:
        for example in dataset.examples:
            if (example[attr_index] == val):
                gain_upper_dataset.examples.append(example)
            else:
                gain_lower_dataset.examples.append(example) 

    if (len(gain_upper_dataset.examples) == 0 or len(gain_lower_dataset.examples) == 0): #Splitting didn't actually split (we tried to split on the max or min of the attribute's range)
        return -1

    attr_entropy += calc_dataset_entropy(gain_upper_dataset, classifier)*len(gain_upper_dataset.examples)/total_examples
    attr_entropy += calc_dataset_entropy(gain_lower_dataset, classifier)*len(gain_lower_dataset.examples)/total_examples
    global idorc
    if not (idorc == 1):
        return entropy - attr_entropy
    else:
        return (entropy - attr_entropy) / (-(len(gain_upper_dataset.examples)/total_examples) * np.log(len(gain_upper_dataset.examples)/total_examples) \
            -(len(gain_lower_dataset.examples)/total_examples) * np.log(len(gain_lower_dataset.examples)/total_examples))

##################################################
# count number of examples with classification "1"
##################################################
def one_count(instances, attributes, classifier):
    count = 0
    class_index = None
    #find index of classifier
    for a in range(len(attributes)):
        if attributes[a] == classifier:
            class_index = a
            break
        else:
            class_index = len(attributes) - 1
    for i in instances:
        if i[class_index] == "1":
            count += 1
    return count

##################################################
# Prune tree
##################################################
def prune_tree(root, node, dataset, best_score):
    # if node is a leaf
    if (node.is_leaf == True):
        # get its classification
        classification = node.classification
        # run validate_tree on a tree with the nodes parent as a leaf with its classification
        node.parent.is_leaf = True
        node.parent.classification = node.classification
        if (node.height < 20):
            new_score = validate_tree(root, dataset)
        else:
            new_score = 0
  
        # if its better, change it
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.classification = None
            return best_score
    # if its not a leaf
    else:
        # prune tree(node.upper_child)
        new_score = prune_tree(root, node.upper_child, dataset, best_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score
        # prune tree(node.lower_child)
        new_score = prune_tree(root, node.lower_child, dataset, new_score)
        # if its now a leaf, return
        if (node.is_leaf == True):
            return new_score

        return new_score

##################################################
# Validate tree
##################################################
def validate_tree(node, dataset):
    total = len(dataset.examples)
    correct = 0
    for example in dataset.examples:
        # validate example
        correct += validate_example(node, example)
    return correct/total

##################################################
# Validate example
##################################################
def validate_example(node, example):
    if (node.is_leaf == True):
        projected = node.classification
        actual = int(example[-1])
        if (projected == actual): 
            return 1
        else:
            return 0
    value = example[node.attr_split_index]
    if node.attr_split_type == True:
        if (value >= node.attr_split_value):
            return validate_example(node.upper_child, example)
        else:
            return validate_example(node.lower_child, example)
    else:
        if (value == node.attr_split_value):
            return validate_example(node.upper_child, example)
        else:
            return validate_example(node.lower_child, example)        
##################################################
# Test example
##################################################
def test_example(example, node, class_index):
    if (node.is_leaf == True):
        return node.classification
    else:
        if node.attr_split_type == True:
            if (example[node.attr_split_index] >= node.attr_split_value):
                return test_example(example, node.upper_child, class_index)
            else:
                return test_example(example, node.lower_child, class_index)
        else:
            if (example[node.attr_split_index] == node.attr_split_value):
                return test_example(example, node.upper_child, class_index)
            else:
                return test_example(example, node.lower_child, class_index)            

##################################################
# Print tree
##################################################
def print_tree(node):
    if (node.is_leaf == True):
        for x in range(node.height):
            print ("\t"),
        print ("Classification: " + str(node.classification))
        return
    for x in range(node.height):
            print ("\t",)
    print ("Split index: " + str(node.attr_split))
    for x in range(node.height):
            print ("\t",)
    print ("Split value: " + str(node.attr_split_value))
    print_tree(node.upper_child)
    print_tree(node.lower_child)

##################################################
# Count the nodes of the tree
##################################################
def count_nodes(node):
    global nodeoftree
    nodeoftree = nodeoftree + 1
    if (node.is_leaf == True):
        return
    count_nodes(node.upper_child)
    count_nodes(node.lower_child)


##################################################
# Print tree in disjunctive normal form
##################################################
def print_disjunctive(node, dataset, dnf_string):
    if (node.parent == None):
        dnf_string = "( "
    if (node.is_leaf == True):
        if (node.classification == 1):
            dnf_string = dnf_string[:-3]
            dnf_string += ") ^ "
            print (dnf_string,)
        else:
            return
    else:
        upper = dnf_string + str(dataset.attributes[node.attr_split_index]) + " >= " + str(node.attr_split_value) + " V "
        print_disjunctive(node.upper_child, dataset, upper)
        lower = dnf_string + str(dataset.attributes[node.attr_split_index]) + " < " + str(node.attr_split_value) + " V "
        print_disjunctive(node.lower_child, dataset, lower)
        return

##################################################
# Necessary functions for ML and JVHW estimator
##################################################

def est_entro_JVHW(samp):
    """Proposed JVHW estimate of Shannon entropy (in bits) of the input sample

    This function returns a scalar JVHW estimate of the entropy of samp when
    samp is a vector, or returns a row vector containing the JVHW estimate of
    each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each column
               of the input matrix. The output data type is double.
    """
    samp = np.asarray(samp)
    samp = samp.astype(np.float)
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    # The order of polynomial is no more than 22 because otherwise floating-point error occurs
    order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
    global poly_entro
    if poly_entro is None:
        poly_entro = sio.loadmat('poly_coeff_entro.mat')['poly_entro']
    coeff = poly_entro[order-1, 0][0]

    f = fingerprint(samp)

    prob = np.arange(1, f.shape[0] + 1) / n

    # Piecewise linear/quadratic fit of c_1
    V1 = np.array([0.3303, 0.4679])
    V2 = np.array([-0.530556484842359, 1.09787328176926, 0.184831781602259])
    f1nonzero = f[0] > 0
    c_1 = np.zeros(wid)

    with np.errstate(divide='ignore', invalid='ignore'):
        if n >= order and f1nonzero.any():
            if n < 200:
                c_1[f1nonzero] = np.polyval(V1, np.log(n / f[0, f1nonzero]))
            else:
                n2f1_small = f1nonzero & (np.log(n / f[0]) <= 1.5)
                n2f1_large = f1nonzero & (np.log(n / f[0]) > 1.5)
                c_1[n2f1_small] = np.polyval(V2, np.log(n / f[0, n2f1_small]))
                c_1[n2f1_large] = np.polyval(V1, np.log(n / f[0, n2f1_large]))

            # make sure nonzero threshold is higher than 1/n
            c_1[f1nonzero] = np.maximum(c_1[f1nonzero], 1 / (1.9 * np.log(n)))

        prob_mat = entro_mat(prob, n, coeff, c_1)

    return np.sum(f * prob_mat, axis=0) / np.log(2)

def entro_mat(x, n, g_coeff, c_1):
    # g_coeff = {g0, g1, g2, ..., g_K}, K: the order of best polynomial approximation,
    K = len(g_coeff) - 1
    thres = 4 * c_1 * np.log(n) / n
    T, X = np.meshgrid(thres, x)
    ratio = np.minimum(np.maximum(2 * X / T - 1, 0), 1)
    q = np.arange(K).reshape((1, 1, K))
    g = g_coeff.reshape((1, 1, K + 1))
    MLE = - X * np.log(X) + 1 / (2 * n)
    polyApp = np.sum(np.concatenate((T[..., None], ((n * X)[..., None]  - q) / (T[..., None] * (n - q))), axis=2).cumprod(axis=2) * g, axis=2) - X * np.log(T)
    polyfail = np.isnan(polyApp) | np.isinf(polyApp)
    polyApp[polyfail] = MLE[polyfail]
    output = ratio * MLE + (1 - ratio) * polyApp
    return np.maximum(output, 0)


def est_entro_MLE(samp):
    """Maximum likelihood estimate of Shannon entropy (in bits) of the input
    sample

    This function returns a scalar MLE of the entropy of samp when samp is a
    vector, or returns a (row-) vector consisting of the MLE of the entropy
    of each column of samp when samp is a matrix.

    Input:
    ----- samp: a vector or matrix which can only contain integers. The input
                data type can be any interger classes such as uint8/int8/
                uint16/int16/uint32/int32/uint64/int64, or floating-point
                such as single/double.
    Output:
    ----- est: the entropy (in bits) of the input vector or that of each
               column of the input matrix. The output data type is double.
    """
    samp = np.asarray(samp)
    samp = samp.astype(np.float)
    samp = formalize_sample(samp)
    [n, wid] = samp.shape
    n = float(n)

    f = fingerprint(samp)
    prob = np.arange(1, f.shape[0] + 1) / n
    prob_mat = - prob * np.log2(prob)
    return prob_mat.dot(f)


def formalize_sample(samp):
    samp = np.array(samp)
    if np.any(samp != np.fix(samp)):
        raise ValueError('Input sample must only contain integers.')
    if samp.ndim == 1 or samp.ndim == 2 and samp.shape[0] == 1:
        samp = samp.reshape((samp.size, 1))
    return samp

def fingerprint(samp):
    """A memory-efficient algorithm for computing fingerprint when wid is
    large, e.g., wid = 100
    """
    wid = samp.shape[1]

    d = np.r_[
        np.full((1, wid), True, dtype=bool),
        np.diff(np.sort(samp, axis=0), 1, 0) != 0,
        np.full((1, wid), True, dtype=bool)
    ]

    f_col = []
    f_max = 0

    for k in range(wid):
        a = np.diff(np.flatnonzero(d[:, k]))
        a_max = a.max()
        hist, _ = np.histogram(a, bins=a_max, range=(1, a_max + 1))
        f_col.append(hist)
        if a_max > f_max:
            f_max = a_max

    return np.array([np.r_[col, [0] * (f_max - len(col))] for col in f_col]).T



##################################################
# main function, organize data and execute functions based on input
# need to account for missing data
##################################################

def main():
    global jvormle
    global idorc
    global eng
    args = str(sys.argv)
    args = ast.literal_eval(args)
    if (len(args) < 2):
        print ("You have input less than the minimum number of arguments. Go back and read README.txt and do it right next time!")
    #elif (args[1][-4:] != ".csv"):
        #print ("Your training file (second argument) must be a .csv!")
    else:
        datafile = args[1]
        dataset = data("")
        if ("-d" in args):
            datatypes = args[args.index("-d") + 1]
        else:
            datatypes = 'datatypes.csv'
        read_data(dataset, datafile, datatypes)
        classifier = dataset.attributes[-1]
        if ("-a" in args):
            arg3 = args[args.index("-a") + 1]
            if (arg3 in dataset.attributes):
                classifier = arg3
        if ("-m" in args):
            methods = args[args.index("-m") + 1]
            if methods == "C4.5":
                idorc = 1
            elif methods == "ID3":
                idorc = 0
            elif methods == "CART":
                idorc = 2
                eng = matlab.engine.start_matlab()
            else:
                print("Methods must be one of the following: C4.5, ID3 or CART.")
                exit(-1)
        if ("-e" in args):
            estimators = args[args.index("-e") + 1]
            if estimators == "JVHW":
                jvormle = 0
            elif estimators == "MLE":
                jvormle = 1
            else:
                print("Estimators must be one of the following: JVHW, MLE.")
                exit(-1)
        dataset.classifier = classifier

        #find index of classifier
        for a in range(len(dataset.attributes)):
            if dataset.attributes[a] == dataset.classifier:
                dataset.class_index = a
                break
            else:
                dataset.class_index = range(len(dataset.attributes))[-1]
                
        unprocessed = copy.deepcopy(dataset)
        class_dict = preprocess2(dataset, None)

        print ("Computing tree...")
        root = compute_tree(dataset, None, classifier) 
        if ("-s" in args):
            print_disjunctive(root, dataset, "")
            print ("\n")
        if ("-v" in args):
            datavalidate = args[args.index("-v") + 1]
            print ("Validating tree...")

            validateset = data(classifier)
            read_data(validateset, datavalidate, datatypes)
            for a in range(len(dataset.attributes)):
                if validateset.attributes[a] == validateset.classifier:
                    validateset.class_index = a
                    break
                else:
                    validateset.class_index = range(len(validateset.attributes))[-1]
            preprocess2(validateset, class_dict)
            best_score = validate_tree(root, validateset)
            all_ex_score = copy.deepcopy(best_score)
            print ("Initial (pre-pruning) validation set score: " + str(100*best_score) +"%")
        global nodeoftree
        nodeoftree = 0
        count_nodes(root)
        print("No. of nodes: " + str(nodeoftree))
        if ("-p" in args):
            if("-v" not in args):
                print ("Error: You must validate if you want to prune")
            else:
                post_prune_accuracy = 100*prune_tree(root, root, validateset, best_score)
                print ("Post-pruning score on validation set: " + str(post_prune_accuracy) + "%")
        if ("-t" in args):
            datatest = args[args.index("-t") + 1]
            testset = data(classifier)
            read_data(testset, datatest, datatypes)
            for a in range(len(dataset.attributes)):
                if testset.attributes[a] == testset.classifier:
                    testset.class_index = a
                    break
                else:
                    testset.class_index = range(len(testset.attributes))[-1]
            print ("Testing model on " + str(datatest))
            for example in testset.examples:
                example[testset.class_index] = '0'
            testset.examples[0][testset.class_index] = '1'
            testset.examples[1][testset.class_index] = '1'
            testset.examples[2][testset.class_index] = '?'
            preprocess2(testset, class_dict)
            b = open('results.csv', 'w')
            a = csv.writer(b)
            for example in testset.examples:
                example[testset.class_index] = test_example(example, root, testset.class_index)
            saveset = testset
            saveset.examples = [saveset.attributes] + saveset.examples
            a.writerows(saveset.examples)
            b.close()
            print ("Testing complete. Results outputted to results.csv")
            
if __name__ == "__main__":
	main()