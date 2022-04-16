import numpy as np

# The original code only outputs the accuracy and the loss.
# Process the file model_output.tsv and calculate precision, recall, and F1 for each class

def precision(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)
    
    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)
    
    tp, fp, tn, fn = get_stats(outputs, labels)
        
    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return tp/(tp + fp)

def recall(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)
    
    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)
    
    tp, fp, tn, fn = get_stats(outputs, labels)
        
    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return tp/(tp + fn)

def f1(outputs, labels):
    p = precision(outputs, labels)
    r = recall(outputs, labels)
    
    return 2 * ((p * r) / (p + r))

def get_stats(outputs, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    # extract stats
    for i, el in enumerate(outputs):
        if labels[i] == 1 and outputs[i] == 1:
            tp = tp + 1
        
        if labels[i] == 0 and outputs[i] == 1:
            fp = fp + 1
        
        if labels[i] == 0 and outputs[i] == 0:
            tn = tn + 1
        
        if labels[i] == 1 and outputs[i] == 0:
            fn = fn + 1
    
    return tp, fp, tn, fn