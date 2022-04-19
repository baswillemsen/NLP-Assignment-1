import numpy as np

# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

# if calculating precision, recall for non-complex class then set complex to False
def precision(outputs, labels, do_preprocess=False, complex=True):    
    # handle baseline code case
    if do_preprocess:
        outputs = preprocess(outputs)
        labels = preprocess(labels)
    else: # handle net class code
        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.ravel()
        # np.argmax gives us the class predicted for each token by the model
        outputs = np.argmax(outputs, axis=1)
    
    tp, fp, tn, fn = get_stats(outputs, labels, complex=complex)
    
    # calculate precision according to formula and handle case where no positives were predicted
    if (tp + fp) == 0:
        return 'N/A'
    else:
        return tp/(tp + fp)

def recall(outputs, labels, do_preprocess=False, complex=True):    
    # handle baseline code case
    if do_preprocess:
        outputs = preprocess(outputs)
        labels = preprocess(labels)
    else: # handle net class code
        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.ravel()
        # np.argmax gives us the class predicted for each token by the model
        outputs = np.argmax(outputs, axis=1)
    
    tp, fp, tn, fn = get_stats(outputs, labels, complex=complex)
    
    # calculate recall according to formula and handle case where no true positives or false negatives were predicted
    if tp + fn == 0:
        return 'N/A'
    else:
        return tp/(tp + fn)

def f1_for_baselines(precision, recall):    
    if precision == 'N/A' or recall == 'N/A':
        return 'N/A'
    else:
        return 2 * ((precision * recall) / (precision + recall))
    
def f1(outputs, labels, do_preprocess=False, complex=True):
    p = precision(outputs, labels, do_preprocess=do_preprocess, complex=complex)
    r = recall(outputs, labels, do_preprocess=do_preprocess, complex=complex)
    
    if p == 'N/A' or r == 'N/A' or (p + r) == 0:
        return 'N/A'
    else:
        return 2 * ((p * r) / (p + r))

# utility method that converts 'N' to 0 and 'C' to 1
def preprocess(input):
    return [1 if x == 'C' else 0 for x in input]

def get_stats(outputs, labels, complex=True):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    # positive result is considered for non-complex prediction 
    pos = 0 # non-complex class (0)
    neg = 1 # complex class (1)
    if complex:
        # positive result is considered for complex prediction
        pos = 1
        neg = 0   
        
    # extract stats
    for i, el in enumerate(outputs):
        if labels[i] == pos and outputs[i] == pos:
            tp = tp + 1
        
        if labels[i] == neg and outputs[i] == pos:
            fp = fp + 1
        
        if labels[i] == neg and outputs[i] == neg:
            tn = tn + 1
        
        if labels[i] == pos and outputs[i] == neg:
            fn = fn + 1
    
    return tp, fp, tn, fn