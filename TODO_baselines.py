# Implement four baselines for the task.
# Majority baseline: always assigns the majority class of the training data
# Random baseline: randomly assigns one of the classes. Make sure to set a random seed and average the accuracy over 100 runs.
# Length baseline: determines the class based on a length threshold
# Frequency baseline: determines the class based on a frequency threshold

# Each baseline returns predictions for the test data. The length and frequency baselines determine a threshold using the development data.
import pandas as pd
from wordfreq import word_frequency
import random
random.seed(0)


# Preprocess: make a dataframe with columns words and labels -----------------------------------------------------------
def preprocess(data_input, data_labels):
    words = ' '.join(data_input).replace('\n', '').split(' ')
    labels = ' '.join(data_labels).replace('\n', '').split(' ')
    df = pd.DataFrame({'words': words, 'labels': labels})
    return df


# Random baseline ------------------------------------------------------------------------------------------------------
def majority_baseline(train_input, train_labels, data_input, data_labels):
    df = preprocess(train_input, train_labels)
    majority_class = df['labels'].value_counts().idxmax()

    df_majority = preprocess(data_input, data_labels)
    predictions = [majority_class] * len(df_majority)
    df_majority['predictions'] = predictions
    accuracy = sum(df_majority['labels'] == df_majority['predictions']) / len(df_majority)

    return accuracy, predictions


# Random baseline ------------------------------------------------------------------------------------------------------
def random_baseline(train_input, train_labels, data_input, data_labels):
    df = preprocess(train_input, train_labels)

    df_random = preprocess(data_input, data_labels)
    predictions = []
    for i in range(len(df_random)):
        if random.random() > 0.5:
            predictions.append('N')
        else:
            predictions.append('C')
    df_random['predictions'] = predictions
    accuracy = sum(df_random['labels'] == df_random['predictions']) / len(df_random)

    return accuracy, predictions


# Length baseline ------------------------------------------------------------------------------------------------------
def length_baseline(train_input, train_labels, data_input, data_labels):
    df = preprocess(train_input, train_labels)

    df_length = preprocess(data_input, data_labels)
    predictions = []
    for word in df_length['words']:
        if len(word) > 8:
            predictions.append('C')
        else:
            predictions.append('N')
    df_length['predictions'] = predictions
    accuracy = sum(df_length['labels'] == df_length['predictions']) / len(df_length)

    return accuracy, predictions


# Frequency baseline ---------------------------------------------------------------------------------------------------
def frequency_baseline(train_input, train_labels, data_input, data_labels):
    df = preprocess(train_input, train_labels)

    df_frequency = preprocess(data_input, data_labels)
    predictions = []
    freq = []
    for word in df_frequency['words']:
        if word_frequency(word, 'en') < 0.000018:
            predictions.append('C')
        else:
            predictions.append('N')
    df_frequency['predictions'] = predictions
    accuracy = sum(df_frequency['labels'] == df_frequency['predictions']) / len(df_frequency)

    return accuracy, predictions


# Run the models -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_path = "data/preprocessed/train/"
    dev_path = "data/preprocessed/val/"
    test_path = "data/preprocessed/test/"

    # Note: this loads all instances into memory. If you work with bigger files in the future, use an iterator instead.
    # Train data -------------------------------------------------------------------------------------------------------
    with open(train_path + "sentences.txt", encoding='utf-8') as train_file:
        train_input = train_file.readlines()
    with open(train_path + "labels.txt", encoding='utf-8') as train_label_file:
        train_labels = train_label_file.readlines()
    # Test data --------------------------------------------------------------------------------------------------------
    with open(test_path + "sentences.txt", encoding='utf-8') as test_file:
        test_input = test_file.readlines()
    with open(test_path + "labels.txt", encoding='utf-8') as test_label_file:
        test_labels = test_label_file.readlines()
    # Dev data ---------------------------------------------------------------------------------------------------------
    with open(dev_path + "sentences.txt", encoding='utf-8') as dev_file:
        dev_input = dev_file.readlines()
    with open(dev_path + "labels.txt", encoding='utf-8') as dev_label_file:
        dev_labels = dev_label_file.readlines()

print('Accuracy on dev: \n')
majority_accuracy_dev, majority_predictions_dev = majority_baseline(train_input, train_labels, dev_input, dev_labels)
random_accuracy_dev, random_predictions_dev = random_baseline(train_input, train_labels, dev_input, dev_labels)
length_accuracy_dev, length_predictions_dev = length_baseline(train_input, train_labels, dev_input, dev_labels)
frequency_accuracy_dev, frequency_predictions_dev = frequency_baseline(train_input, train_labels, dev_input, dev_labels)
print(f'{majority_accuracy_dev:.2f}', f'{random_accuracy_dev:.2f}', f'{length_accuracy_dev:.2f}', f'{frequency_accuracy_dev:.2f}')

print('Accuracy on test: \n')
majority_accuracy_test, majority_predictions_test = majority_baseline(train_input, train_labels, test_input, test_labels)
random_accuracy_test, random_predictions_test = random_baseline(train_input, train_labels, test_input, test_labels)
length_accuracy_test, length_predictions_test = length_baseline(train_input, train_labels, test_input, test_labels)
frequency_accuracy_test, frequency_predictions_test = frequency_baseline(train_input, train_labels, test_input, test_labels)
print(f'{majority_accuracy_test:.2f}', f'{random_accuracy_test:.2f}', f'{length_accuracy_test:.2f}', f'{frequency_accuracy_test:.2f}')