import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.nn import MultiheadAttention, Dropout, LayerNorm, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torchmetrics
from multiprocessing import Pool
from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, 
    BertPreTrainedModel, BertModel, BertConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from text_processing1 import lemmatize_sentences
from typing import Optional, Union, Tuple

# Optional downloads for spaCy and NLTK:
# spacy.cli.download("en_core_web_sm")
# nltk.download('stopwords')


def plot_confusion_matrix(aspect_list, real_aspect, reviews_assigned_topic, all_index,
                          normalize='true'):  # for same color scale in each matrix

    """
    Plots confusion matrices for each aspect, comparing real and predicted labels.

    Args:
        aspect_list (list): List of aspects to evaluate.
        real_aspect (dict): Indices with real aspect labels. Dict of lists
        reviews_assigned_topic (dict): Indices assigned to predicted aspects. Dict of lists
        all_index (list): List of all review indices.
        normalize (str): Normalization method ('true', 'pred', or 'all'). Default is 'true'.

    Returns:
        None: Displays confusion matrices for each aspect with normalization.
    """

    # Determine overall min and max values for normalization across all plots
    overall_min = 0  # Adjust this based on your data
    overall_max = 1  # Adjust this based on your data

    # Plotting confusion matrices
    fig, axes = plt.subplots(3, 4, figsize=(25, 15))  # Adjust the figsize as needed
    plt.suptitle('Confusion Matrix for Aspects, normalized on {} labels'.format(normalize))

    for ax, aspect in zip(axes.flatten(), aspect_list):
        ax.set_title(aspect)

        true_aspect = ['IN' if x in real_aspect[aspect] else 'NOT IN' for x in all_index]
        pred_aspect = ['IN' if x in reviews_assigned_topic[aspect] else 'NOT IN' for x in all_index]
        # count number of real and predicted reviews for each aspect
        unique_values, counts = np.unique(true_aspect, return_counts=True)
        count_dict = dict(zip(unique_values, counts))
        print(count_dict)
        print(aspect, 'true:', count_dict)
        unique_values, counts = np.unique(pred_aspect, return_counts=True)
        count_dict = dict(zip(unique_values, counts))
        print(aspect, 'pred:', count_dict)

        # Compute confusion matrix
        cm = confusion_matrix(true_aspect, pred_aspect, normalize=normalize)

        # Display confusion matrix with consistent color scale
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=overall_min, vmax=overall_max)
        fig.colorbar(im, ax=ax)

        # Show all ticks and label them with respective list entries
        ax.set_xticks(np.arange(len(['IN', 'NOT IN'])))
        ax.set_yticks(np.arange(len(['IN', 'NOT IN'])))
        ax.set_xticklabels(['IN', 'NOT IN'])
        ax.set_yticklabels(['IN', 'NOT IN'])

        # Loop over data dimensions and create text annotations
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()  # Adjust subplot parameters to give room for titles and labels
    plt.show()


def replace_topics_with_aspects(pred_aspect_index, zip_list):
    """
    Replaces topics in the predicted aspect index with corresponding aspects using a mapping.

    Args:
        pred_aspect_index (dict): Dictionary of predicted topics indexed by keys.
        zip_list (list): List of tuples mapping topics to aspects.

    Returns:
        dict: Dictionary with topics replaced by aspects.
    """

    # Initialize the result dictionary
    replaced_dict = {}

    # Create a mapping from the first value to the third value
    mapping = {item[0]: item[2] for item in zip_list}

    # Iterate through the dictionary
    for key, values in pred_aspect_index.items():
        # Replace elements based on the mapping, filter out non-matching elements, and suppress duplicates
        # replaced_values = list({topic_to_aspect[value] for value in values if value in topic_to_aspect})
        replaced_dict[key] = [mapping[value] for value in values if value in mapping]
        # if replaced_values:  # Only add if there are matching aspects
        #   replaced_dict[key] = replaced_values
    return replaced_dict


def compute_hamming_loss(pred_aspect_index, real_aspect_index, aspect_list):
    """
    Computes the Hamming Loss between predicted and real aspects.

    Args:
        pred_aspect_index (dict): Predicted aspects indexed by keys.
        real_aspect_index (dict): Real aspects indexed by keys.
        aspect_list (list): List of all possible aspects.

    Returns:
        float: Hamming Loss value.
    """

    total_labels = 0
    incorrect_labels = 0

    # Get all unique indices
    all_indices = set(pred_aspect_index.keys()).union(set(real_aspect_index.keys()))
    for index in all_indices:
        # print(index)
        # Get predicted and real aspects as sets
        pred_aspects = set(pred_aspect_index.get(index, []))
        real_aspects = set(real_aspect_index.get(index, []))
        # print(pred_aspects)
        # print(real_aspects)

        # Compute symmetric difference
        differences = pred_aspects.symmetric_difference(real_aspects)
        # print(differences)

        # Update counts
        incorrect_labels += len(differences)
        total_labels += len(aspect_list)

    # Calculate Hamming Loss
    hamming_loss = incorrect_labels / total_labels if total_labels != 0 else 0
    return hamming_loss


def make_topic_aspect_link(topic_model, candidate_categories, category_aspect_correspondance):
    """
    Links topic names from the topic model to corresponding aspects.

    Args:
        topic_model: A topic model object with topic information.
        candidate_categories (list): List of category names to match.
        category_aspect_correspondance (dict): Mapping of category names to aspects.

    Returns:
        list: A list of tuples with (Topic, Name, Corresponding Aspect).
    """

    # Get topic information and replace underscores
    df = topic_model.get_topic_info().copy()
    df['Name'] = df['Name'].apply(lambda x: x.replace('_', ''))
    df['Name'] = df['Name'].apply(lambda x: re.sub(r'\d+', '', x))
    zip_list = []

    def apply_link(row):
        # Check if the name is in candidate aspects
        if row['Name'] in candidate_categories:
            to_append = (row['Topic'], row['Name'], category_aspect_correspondance[row['Name']])
            zip_list.append(to_append)

    # Apply the function to each row
    df.apply(lambda row: apply_link(row), axis=1)

    return zip_list


def apply_evaluate_btp(dict_reviews, list_sentences, topic_model, real_labels, aspect_list, candidate_categories,
                       category_aspect_correspondance, outliers_reduction=None):

    """
    Applies a topic model to evaluate predictions against real labels, computes metrics, and generates a scores dataframe.

    Args:
        dict_reviews (dict): Mapping of reviews to indices.
        list_sentences (list): Sentences to be processed by the topic model.
        topic_model: Pretrained topic model to assign topics.
        real_labels (DataFrame): True labels for evaluation.
        aspect_list (list): List of aspects to evaluate.
        candidate_categories (list): Categories used in the topic model.
        category_aspect_correspondance (dict): Maps categories to aspects.
        outliers_reduction (str, optional): Strategy to reduce outliers. Defaults to None.

    Returns:
        tuple: Contains predicted topics, probabilities, Hamming loss, the topic model, assigned topics, and a dataframe of evaluation scores.
    """

    # candidate_categories must correspond to the candidate categories that were given for
    # representation_model = ZeroShotClassification(candidate_categories) in the training of topic_model
    # aspect_list is a sublist of full_aspect_list, and aspects of category_aspect_correspondance should be restricted to those in aspect_list

    # Create a reverse index dictionary
    reverse_index_dict = {}
    for index, values in dict_reviews.items():
        for value in values:
            reverse_index_dict[value] = index

    # Use the reverse index dictionary for lookup
    reverse_lbl = [reverse_index_dict.get(string) for string in list_sentences]

    # Apply topic model to get topics and probabilities
    topics_lbl, probs_lbl = topic_model.transform(list_sentences)
    if outliers_reduction != None:
        topics_lbl = topic_model.reduce_outliers(list_sentences, topics_lbl, strategy=outliers_reduction)

    # Initialize dictionaries to store assigned reviews and real aspects
    reviews_assigned_topic = {}
    real_aspect = {}
    real_aspect_index = {}
    pred_aspect_index = {}

    # Loop through each aspect and its corresponding topic index
    zip_list = make_topic_aspect_link(topic_model, candidate_categories, category_aspect_correspondance)
    # print('zip_list:', zip_list)
    # initialize reviews_assigned_topic for each aspect
    for aspect in aspect_list:
        reviews_assigned_topic[aspect] = []
    for topic_index, category, aspect in zip_list:
        index_sentences_topic = [index for index, value in enumerate(topics_lbl) if value == topic_index]

        # IDs of the test set that are assigned to the current topic_index
        indices_to_append = [reverse_lbl[i] for i in index_sentences_topic]
        reviews_assigned_topic[aspect].extend(indices_to_append)
        # print(aspect, reviews_assigned_topic[aspect])
    for aspect in aspect_list:
        # IDs of the reviews that are labeled as the current aspect
        real_aspect[aspect] = list(real_labels[(~(real_labels[aspect].isna()))]['Index'])
        # print(aspect)
        # print(list(real_labels[(~(real_labels[aspect].isna()))]['Index']))
    # print(reviews_assigned_topic)
    for index in real_labels['Index']:
        df = real_labels.set_index('Index')
        var_df = df.loc[index][aspect_list].isna()
        real_aspect_index[index] = sorted(var_df[var_df].index.tolist())
        # for each index, aspects that are predicted
        pred_aspect_index[index] = [topics_lbl[pos] for pos, index_reverse in enumerate(reverse_lbl) if
                                    index_reverse == index]
    # print(pred_aspect_index)
    pred_aspect_index = replace_topics_with_aspects(pred_aspect_index, zip_list)

    # All index of reviews
    all_index = list(real_labels['Index'])

    plot_confusion_matrix(aspect_list, real_aspect, reviews_assigned_topic, all_index, normalize='true')
    plot_confusion_matrix(aspect_list, real_aspect, reviews_assigned_topic, all_index, normalize='pred')
    # plot_confusion_matrix(aspect_list, real_aspect, reviews_assigned_topic, all_index, normalize='all')
    # print(reverse_lbl)
    # print('pred_aspect_index')
    # print(pred_aspect_index)
    # print('real_aspect_index')
    # print(real_aspect_index)

    hamming_loss = compute_hamming_loss(pred_aspect_index, real_aspect_index, aspect_list)

    # Compute F1 score and accuracy for each aspect
    f1_score_dict = {}
    accuracy_score_dict = {}
    recall_score_dict = {}
    precision_score_dict = {}
    f1_score_dict_skl = {}

    for aspect in aspect_list:
        y_true = [(index in real_aspect[aspect]) for index in all_index]
        y_pred = [(index in reviews_assigned_topic[aspect]) for index in all_index]

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Store metrics in dictionaries
        f1_score_dict[aspect] = f1
        accuracy_score_dict[aspect] = accuracy
        recall_score_dict[aspect] = recall
        precision_score_dict[aspect] = precision
        f1_score_dict_skl[aspect] = f1_score(y_true, y_pred)

    f1_score_df = pd.DataFrame.from_dict(f1_score_dict, orient='index', columns=['F1 Score'])
    f1_score_df.reset_index(inplace=True)

    accuracy_score_df = pd.DataFrame.from_dict(accuracy_score_dict, orient='index', columns=['Accuracy Score'])
    accuracy_score_df.reset_index(inplace=True)

    recall_score_df = pd.DataFrame.from_dict(recall_score_dict, orient='index', columns=['Recall'])
    recall_score_df.reset_index(inplace=True)

    precision_score_df = pd.DataFrame.from_dict(precision_score_dict, orient='index', columns=['Precision'])
    precision_score_df.reset_index(inplace=True)

    f1_score_df_skl = pd.DataFrame.from_dict(f1_score_dict_skl, orient='index', columns=['F1 score skl'])
    f1_score_df_skl.reset_index(inplace=True)

    scores_df = accuracy_score_df.merge(f1_score_df, on='index') \
        .merge(recall_score_df, on='index') \
        .merge(precision_score_df, on='index') \
        .merge(f1_score_df_skl, on='index')

    scores_df.rename(columns={'index': 'Aspect'}, inplace=True)

    return topics_lbl, probs_lbl, hamming_loss, topic_model, reviews_assigned_topic, scores_df


def plot_confusion_matrix_keywords(aspect_keywords, full_test_set, real_labels, normalize='true',
                                   average_f1_score='binary'):  # for same color scale in each matrix

    """
    Plot confusion matrices for different aspects based on keyword predictions (Lemma Keywords Detection results).

    This function generates and displays confusion matrices for aspect keyword detection from a set of reviews.
    It also calculates various performance metrics (F1 score, accuracy, recall, precision) for each aspect.

    Args:
        aspect_keywords (dict):
            A dictionary where keys are aspect names and values are lists of keywords associated with those aspects.
        full_test_set (pd.DataFrame):
            A DataFrame containing the full test set, which should have 'Index' and 'Review' columns.
        real_labels (pd.Series):
            A Series containing the true labels for each review, indexed by review IDs.
        normalize (str, optional):
            Normalization mode for the confusion matrix ('true', 'pred', or None). Defaults to 'true'.
        average_f1_score (str, optional):
            Method for averaging the F1 score ('binary' or 'macro'). Defaults to 'binary'.

    Returns:
        pd.DataFrame:
            A DataFrame containing the calculated scores (F1 score, accuracy, recall, precision) for each aspect.

    Notes:
        - The function assumes that the input DataFrame and Series are properly formatted and contain relevant data.
        - Confusion matrices are plotted using a consistent color scale for comparison across aspects.
    """

    # Determine overall min and max values for normalization across all plots
    overall_min = 0  # Adjust this based on your data
    overall_max = 1  # Adjust this based on your data

    # Plotting confusion matrices
    fig, axes = plt.subplots(3, 4, figsize=(25, 15))  # Adjust the figsize as needed
    plt.suptitle('Confusion Matrix for Aspects, normalized on {} labels'.format(normalize))

    def find_aspect_indices(df, aspect_keywords):
        df_copy = df.set_index('Index')
        aspect_indices = {}
        for aspect, keywords in aspect_keywords.items():
            if keywords:  # Only proceed if there are keywords
                indices = df_copy[df_copy['Review'].str.contains('|'.join(keywords), case=False)].index.tolist()
            else:
                indices = []  # No indices for empty keyword lists
            aspect_indices[aspect] = indices
        return aspect_indices

    # Get the aspect indices
    aspect_indices = find_aspect_indices(full_test_set, aspect_keywords)

    # All reviews IDs
    all_index = full_test_set['Index'].unique()
    # print(all_index)

    real_aspect = {}

    f1_score_dict = {}
    f1_score_dict_skl = {}
    accuracy_score_dict = {}
    recall_score_dict = {}
    precision_score_dict = {}

    for ax, aspect in zip(axes.flatten(), aspect_keywords.keys()):  # max 8 aspects!!!
        ax.set_title(aspect)

        real_aspect[aspect] = list(real_labels[(~(real_labels[aspect].isna()))]['Index'])
        # print(real_aspect[aspect])
        # print(aspect_indices[aspect])
        true_aspect = ['IN' if x in real_aspect[aspect] else 'NOT IN' for x in all_index]
        pred_aspect = ['IN' if x in aspect_indices[aspect] else 'NOT IN' for x in all_index]

        # count number of real and predicted reviews for each aspect
        unique_values, counts = np.unique(true_aspect, return_counts=True)
        count_dict = dict(zip(unique_values, counts))
        # print(count_dict)
        print(aspect, 'true:', count_dict)
        unique_values, counts = np.unique(pred_aspect, return_counts=True)
        count_dict = dict(zip(unique_values, counts))
        print(aspect, 'pred:', count_dict)

        # Compute F1 and accuracy score for each aspect
        # Map 'IN' and 'NOT IN' to numeric values
        true_aspect_numeric = [1 if label == 'IN' else 0 for label in true_aspect]
        pred_aspect_numeric = [1 if label == 'IN' else 0 for label in pred_aspect]

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_aspect_numeric, pred_aspect_numeric).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Store metrics in dictionaries
        f1_score_dict[aspect] = f1
        accuracy_score_dict[aspect] = accuracy
        recall_score_dict[aspect] = recall
        precision_score_dict[aspect] = precision
        f1_score_dict_skl[aspect] = f1_score(true_aspect_numeric, pred_aspect_numeric)

        cm = confusion_matrix(true_aspect, pred_aspect, normalize=normalize)

        # Display confusion matrix with consistent color scale
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=overall_min, vmax=overall_max)
        fig.colorbar(im, ax=ax)

        # Show all ticks and label them with respective list entries
        ax.set_xticks(np.arange(len(['IN', 'NOT IN'])))
        ax.set_yticks(np.arange(len(['IN', 'NOT IN'])))
        ax.set_xticklabels(['IN', 'NOT IN'])
        ax.set_yticklabels(['IN', 'NOT IN'])

        # Loop over data dimensions and create text annotations
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    f1_score_df = pd.DataFrame.from_dict(f1_score_dict, orient='index', columns=['F1 Score'])
    f1_score_df.reset_index(inplace=True)

    accuracy_score_df = pd.DataFrame.from_dict(accuracy_score_dict, orient='index', columns=['Accuracy Score'])
    accuracy_score_df.reset_index(inplace=True)

    recall_score_df = pd.DataFrame.from_dict(recall_score_dict, orient='index', columns=['Recall'])
    recall_score_df.reset_index(inplace=True)

    precision_score_df = pd.DataFrame.from_dict(precision_score_dict, orient='index', columns=['Precision'])
    precision_score_df.reset_index(inplace=True)

    f1_score_df_skl = pd.DataFrame.from_dict(f1_score_dict_skl, orient='index', columns=['F1 score skl'])
    f1_score_df_skl.reset_index(inplace=True)

    scores_df = accuracy_score_df.merge(f1_score_df, on='index') \
        .merge(recall_score_df, on='index') \
        .merge(precision_score_df, on='index') \
        .merge(f1_score_df_skl, on='index')

    scores_df.rename(columns={'index': 'Aspect'}, inplace=True)
    plt.tight_layout()  # Adjust subplot parameters to give room for titles and labels
    plt.show()

    return scores_df


#def remove_stop_words1(df, colname):
#    df_copy = df.copy()
#    stop_words = stopwords.words('english')
#    df_copy[colname] = df_copy[colname].apply(lambda x: [word for word in x if word not in stop_words])
#    return df_copy


def remove_stop_words2(df, colname):
    """
    removes stop words from data according to spacy 'en_core_web_sm'
    :param df (pd.DataFrame): data
    :param colname (string): name of column containing the text from which stop word will be removed
    :return 'pd.DataFrame): cleaned df
    """
    # common_words = ["airline", "flight", "airport", "fly", "plane"]
    df_copy = df.copy()

    # Load the French language model
    nlp = spacy.load('en_core_web_sm')

    df_copy[colname] = df_copy[colname].apply(
        lambda x: [token.text for token in nlp(x) if token.pos_ in ('NOUN', 'VERB', 'PUNCT')])
    # print(list(df_copy[colname]))
    df_copy[colname] = df_copy[colname].apply(lambda x: ' '.join(x))
    # print(list(df_copy[colname]))
    df_copy[colname] = df_copy[colname].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])
    # print(list(df_copy[colname]))
    # df_copy[colname] = df_copy[colname].apply(lambda x: [word for word in x if word not in common_words])
    # print(list(df_copy[colname]))
    df_copy[colname] = df_copy[colname].apply(lambda x: ' '.join(x))
    return df_copy


def split_by_sentence(df):
    """
    Split reviews into sentences.

    Args:
        df (pd.DataFrame): DataFrame containing reviews with an 'Index' and 'Review' column.

    Returns:
        dict: A dictionary where keys are indices and values are lists of sentences.
        list: A flat list of all sentences extracted from the reviews.
    """

    dict_reviews = {}
    list_sentences = []
    df_copy = df.copy()
    df_copy.set_index('Index', inplace=True)
    for index in df_copy.index:
        dict_reviews[index] = df_copy.loc[index]['Review'].split('.')[:-1]
        list_sentences += dict_reviews[index]
    return dict_reviews, list_sentences


def split_and_join_sentences2(df, n, coma=None):
    """
    Split reviews into sentences and combine them into groups.

    Args:
        df (pd.DataFrame): DataFrame containing reviews with an 'Index' and 'Review' column.
        n (int): Maximum number of sentences to combine.
        coma (bool, optional): If True, split by commas as well. Defaults to None.

    Returns:
        dict: A dictionary where keys are indices and values are lists of combined sentences.
        list: A flat list of all combined sentences.
    """

    dict_reviews = {}
    all_combinations = []
    df_copy = df.copy()
    df_copy.set_index('Index', inplace=True)

    for index in df_copy.index:
        if coma != None:
            # Split the review into sentences based on periods and commas
            sentences = re.split(r'[.,]', df_copy.loc[index]['Review'])
            # Remove any empty strings and white spaces
        else:
            # Split the review into sentences
            sentences = df_copy.loc[index]['Review'].split('.')
        # Remove any empty strings and white spaces
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Combine sentences up to n sentences
        combined_sentences = []
        max_group_size = min(n, len(sentences))  # Ensure we do not exceed the length of the sentences or n
        for group_size in range(1, max_group_size + 1):
            for i in range(len(sentences) - group_size + 1):
                # Join 'group_size' number of sentences starting at index 'i'
                combined_sentence = '. '.join(sentences[i:i + group_size]) + '.'
                combined_sentences.append(combined_sentence)

        dict_reviews[index] = combined_sentences
        all_combinations.extend(combined_sentences)

    return dict_reviews, all_combinations



def split_and_join_sentences(df, n, coma=None):

    """
    Splits reviews into sentences and combines them into groups of up to 'n' sentences.

    Args:
        df (pd.DataFrame): DataFrame containing reviews with 'Index' and 'Review' columns.
        n (int): Maximum number of sentences to combine.
        coma (bool, optional): Whether to split sentences based on commas in addition to other punctuation.

    Returns:
        dict: Dictionary of combined sentences for each review indexed by 'Index'.
        list: List of all possible sentence combinations across reviews.
    """

    dict_reviews = {}
    all_combinations = []
    df_copy = df.copy()
    df_copy.set_index('Index', inplace=True)

    for index in df_copy.index:
        if coma is not None:
            # Split the review into sentences based on periods, commas, exclamation marks, and question marks
            sentences = re.split(r'[.,!?]', df_copy.loc[index]['Review'])
        else:
            # Split the review into sentences based on periods, exclamation marks, and question marks
            sentences = re.split(r'[.!?]', df_copy.loc[index]['Review'])

        # Remove any empty strings and white spaces
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Combine sentences up to n sentences
        combined_sentences = []
        max_group_size = min(n, len(sentences))  # Ensure we do not exceed the length of the sentences or n
        for group_size in range(1, max_group_size + 1):
            for i in range(len(sentences) - group_size + 1):
                # Join 'group_size' number of sentences starting at index 'i'
                combined_sentence = '. '.join(sentences[i:i + group_size]) + '.'
                combined_sentences.append(combined_sentence)

        dict_reviews[index] = combined_sentences
        all_combinations.extend(combined_sentences)

    return dict_reviews, all_combinations


def aspect_extraction_keywords(aspect_keywords, data_train, training=True,
                               real_labels=None):  # specify real_labels if training=False
    """
    Performs Lemma Keywords Detection (AE) based on aspect_keywords dict

    Args:
        aspect_keywords (dict): Dict of lemma keywords related to each aspect
            e.g.: aspect_keywords_main_compare = {'food and beverages': ['food', 'beverage', 'meal', 'snack'],
             'entertainment': ['entertainment',
              'movie',
              'screen',
              'wifi',
              'headphone',
              'TV',
              'music',
              'IFE',
              'video',
              'internet',
              'wi-fi',
              'connectivity'],
            'on ground services': [
                    'check-in', 'lounge', 'counter', 'luggage', 'ground staff', 'ground crew', 'check in', 'gate'
                ],
             'delay': ['delay', 'late', 'punctuality'],
             'crew flight': ['attendant', 'steward', 'FA', 'crew', 'hostess', 'staff'],
             'seat comfort': ['legroom', 'armrest', 'recline', 'width', 'seat']}

        data_train (pd.DataFrame): Dataset with review data.
        training (bool, optional): Defaults to True. Uses skytrax ratings if True. Hand-made provided labels (real_labels) if False.
        real_labels (pd.DataFrame, optional): Sentiment labels for non-training mode.

    Returns:
        aspect_sentences (dict): nested dict. For each aspect, a dict with sentences recognized by Lemma Keyword Detection (keys) and their respective review index (values)
        dict_reviews (dict): nested dict. For each aspect, a dict with sentences recognized by Lemma Keyword Detection (keys) and their respective label (0 in negative or 1 if positive)
    """

    if training == True:
        # Restrict aspects to those present in skytrax ratings (both)
        def filter_dict_by_keys(input_dict, allowed_keys):
            return {key: value for key, value in input_dict.items() if key in allowed_keys}

        allowed_keys = ['seat comfort', 'crew flight', 'food and beverages', 'on ground services', 'entertainment']
        aspect_keywords = filter_dict_by_keys(aspect_keywords, allowed_keys)

        def classify(x):
            if x in [1, 2, 3]:
                return int(0)  # NEG
            elif x in [4, 5]:
                return int(1)  # POS
            else:
                return x

        # Applying the function to all columns
        data_train[allowed_keys] = data_train[allowed_keys].applymap(classify)

    def find_aspect_indices(df, aspect_keywords):
        aspect_sentences = {}
        dict_reviews, list_sentences = split_and_join_sentences(df, n=1)
        sentences_df = pd.DataFrame(list_sentences, columns=['Sentence'])

        print('ok')
        # Lemmatize each sentence in the DataFrame using multiprocessing
        sentences_df['Lemmatized_Sentence'] = lemmatize_sentences(sentences_df['Sentence'].tolist())
        print('ok')

        for aspect, keywords in aspect_keywords.items():
            if keywords:  # Only proceed if there are keywords
                # Join keywords with '|' to create a regex pattern
                pattern = '|'.join(keywords)
                # Filter sentences that contain any of the keywords in their lemmatized form
                sentences = sentences_df[sentences_df['Lemmatized_Sentence'].str.contains(pattern, case=False)][
                    'Sentence'].tolist()
            else:
                sentences = []  # No indices for empty keyword lists
            aspect_sentences[aspect] = sentences

        return aspect_sentences, dict_reviews

    # Get the aspect indices and sentence-to-review mapping
    aspect_sentences, dict_reviews = find_aspect_indices(data_train, aspect_keywords)

    # Create a reverse index dictionary
    reverse_index_dict = {}
    for index, values in dict_reviews.items():
        for value in values:
            reverse_index_dict[value] = index

    pred_sentences_link_reviews = {}
    AE_pred_label = {}
    for aspect in aspect_keywords.keys():
        pred_sentences_link_reviews[aspect] = {}
        AE_pred_label[aspect] = {}
        for sentence in aspect_sentences[aspect]:
            index = reverse_index_dict[sentence]
            # Link detected sentence and review index for each aspect
            pred_sentences_link_reviews[aspect][sentence] = index
            if training == True:
                AE_pred_label[aspect][sentence] = data_train[data_train['Index'] == index][aspect].values[0]
            elif training == False:
                # Link detected sentence and review aspect sentiment classification real label
                AE_pred_label[aspect][sentence] = real_labels[real_labels['Index'] == index][aspect].values[0]
            else:
                raise TypeError('training should be True of False')

    return pred_sentences_link_reviews, AE_pred_label


def analyze_list_of_sentences_vader(sentences):
    """Analyze a list of sentences using the VADER sentiment analysis tool.
    Used in evaluate_sentiment_classification_vader function.
    This function computes sentiment scores for each sentence using the
    SentimentIntensityAnalyzer from the VADER sentiment analysis library.

    Args:
        sentences (list of str): A list of sentences to analyze for sentiment.

    Returns:
        dict: A dictionary where the keys are the input sentences and the
              values are their corresponding sentiment scores (compound score).
    """

    analyzer = SentimentIntensityAnalyzer()
    results = {}

    for sentence in sentences:
        sentiment_score = analyzer.polarity_scores(sentence)
        results[sentence] = sentiment_score['compound']

    return results


def evaluate_sentiment_classification_vader(data, aspect, result, pred_sentences_link_reviews):
    """Evaluate the performance of sentiment classification using VADER predictions.

    This function compares the predicted sentiment labels from VADER against
    the true labels from the dataset, calculating accuracy and visualizing
    results with confusion matrices.

    Args:
        data (pd.DataFrame): The original dataset containing true labels.
        aspect (str): The specific aspect of the data being evaluated.
        result (dict): A dictionary of sentences with VADER's predicted sentiment
                       scores, where keys are sentences and values are predictions.
        pred_sentences_link_reviews (dict): A dictionary linking sentences to
                                             their respective review indices.

    Returns:
        float: The accuracy of the sentiment classification predictions.
    """

    # result: result = analyze_list_of_sentences(training_sentences). dict where keys are sentences and values are predictions
    predicted_labels_vader = list(result.values())
    sentences_vader = list(result.keys())

    # link each prediction to the index. Group together the predictions for every index (in case several sentences from a same review)
    # average them, compare to true labels
    test_link_dict = {}
    for enum, prediction in enumerate(predicted_labels_vader):
        index = pred_sentences_link_reviews[aspect][sentences_vader[enum]]
        if index in test_link_dict.keys():
            test_link_dict[index].append(prediction)
        else:
            test_link_dict[index] = [prediction]
    # print(test_link_dict)

    # predictions
    test_final_pred_dict = {}
    for key in test_link_dict.keys():
        test_final_pred_dict[key] = np.mean(test_link_dict[key])
    test_final_pred_df = pd.DataFrame.from_dict(test_final_pred_dict, orient='index', columns=['pred']).reset_index()
    test_final_pred_df.rename(columns={'index': 'Index'}, inplace=True)
    test_final_pred_df['pred'] = np.where(test_final_pred_df['pred'] <= 0, 0, 1)
    test_final_pred_df

    # true labels
    list_indices = [key for key in test_link_dict.keys()]
    df_true_labels = data[data['Index'].isin(list_indices)][['Index', aspect]]
    df_true_labels[aspect] = np.where(df_true_labels[aspect] <= 3, 0, 1)
    df_true_labels

    df_calc_score = df_true_labels.merge(test_final_pred_df, on='Index')
    df_calc_score
    accuracy = (df_calc_score[aspect] == df_calc_score['pred']).mean()

    ConfusionMatrixDisplay.from_predictions(df_calc_score[aspect], df_calc_score['pred'], normalize='true',
                                            cmap='Blues')
    plt.title('{}, normalized on true labels'.format(aspect))
    plt.show()
    ConfusionMatrixDisplay.from_predictions(df_calc_score[aspect], df_calc_score['pred'], normalize='pred', cmap='Reds')
    plt.title('{}, normalized on predicted labels'.format(aspect))
    plt.show()
    return accuracy


def bert_training(data_training, candidate_aspects, random_state, min_prob, nb_sentences=1):
    """
    Trains a BERTopic model with optional zero-shot classification.

    Args:
        data_training (pd.DataFrame): Input data for training.
        candidate_aspects (list): List of aspects for zero-shot classification.
        random_state (int): Random seed for reproducibility.
        min_prob (float): Minimum probability threshold for zero-shot classification.
        nb_sentences (int, optional): Number of sentences to join before topic modeling. Defaults to 1.

    Returns:
        BERTopic: The trained BERTopic model.
        list: List of topics generated from the model.
    """

    dict_reviews, list_sentences = split_and_join_sentences(data_training, n=nb_sentences, coma=None)
    docs = list_sentences

    # Train BERTopic
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=random_state)
    model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)
    representation_model = ZeroShotClassification(candidate_aspects, model=model, min_prob=min_prob)

    topic_model_base = BERTopic(vectorizer_model=vectorizer_model, calculate_probabilities=False,
                                    umap_model=umap_model, representation_model=representation_model)

    topics_base, _ = topic_model_base.fit_transform(docs)
    return topic_model_base, topics_base


# Convert the dictionaries to a DataFrame
def concat_scores(f1_score_dict, accuracy_score_dict):
    """Convert score from function evaluate_score_bertopic dictionaries into a merged DataFrame.

    Args:
        f1_score_dict (dict): Dictionary containing F1 scores.
        accuracy_score_dict (dict): Dictionary containing accuracy scores.

    Returns:
        pd.DataFrame: Merged DataFrame with F1 scores and accuracy scores.
    """

    f1_score_df = pd.DataFrame.from_dict(f1_score_dict, orient='index', columns=['F1 Score'])
    f1_score_df.reset_index(inplace=True)
    accuracy_score_df = pd.DataFrame.from_dict(accuracy_score_dict, orient='index', columns=['Accuracy Score'])
    accuracy_score_df.reset_index(inplace=True)
    scores_dict = accuracy_score_df.merge(f1_score_df, on='index')
    return scores_dict


def evaluate_score_bertopic(testing_data, list_topic_model, real_labels, aspect_list, candidate_aspects,
                            category_aspect_correspondance, outliers_reduction=None,
                            nb_sentences=1):
    """Evaluate scores of a list of BERTopic models.

    Args:
        testing_data (DataFrame): The testing dataset.
        list_topic_model_base (list): List of BERTopic models to evaluate.
        real_labels (DataFrame): Actual labels of the testing data.
        aspect_list (list): List of aspects to consider.
        candidate_aspects (list): List of candidate aspects for matching.
        category_aspect_correspondance (dict): Dictionary mapping 'candidate_aspects' to aspects.
        average_f1_score (str, optional): F1 score averaging method. Defaults to 'weighted'.
        outliers_reduction (str, optional): Method to reduce outliers. Defaults to None.
        nb_sentences (int, optional): Number of sentences to join. Defaults to 1.

    Returns:
        tuple: A tuple containing the final DataFrame with scores and the mean scores DataFrame.
    """

    # dict_reviews_lbl, list_sentences_lbl = split_by_sentence(testing_data) #testing_data = full_test_set_2 usually
    dict_reviews_lbl, list_sentences_lbl = split_and_join_sentences(testing_data, n=nb_sentences, coma=None)
    final_scores_df = pd.DataFrame()
    for topic_model_base in list_topic_model:
        topics_lbl, probs_lbl, hamming_loss, topic_model, reviews_assigned_topic, scores_df = apply_evaluate_btp(
            dict_reviews_lbl, list_sentences_lbl, topic_model_base, real_labels, aspect_list, candidate_aspects,
            category_aspect_correspondance, outliers_reduction=outliers_reduction)
        # scores_df = concat_scores(f1_score_dict, accuracy_score_dict)
        final_scores_df = pd.concat([final_scores_df, scores_df], axis=0)
    print('final_scores_df', final_scores_df)
    print('groupby', final_scores_df.groupby(final_scores_df.index))
    mean_df = final_scores_df.groupby('Aspect').mean()
    return final_scores_df, mean_df


def aspect_extraction_keywords(aspect_keywords, data_train, skytrax_labels=True, real_labels=None,
                               allowed_keys=['seat comfort', 'crew flight', 'food and beverages', 'delay', 'entertainment']):

    """Extract aspect-based sentences from reviews based on specified keywords (Aspect Extraction, Lemma Keywords Detection)

    Args:
        aspect_keywords (dict): Dictionary mapping aspects to keywords.
        data_train (DataFrame): Training data containing reviews and aspect ratings.
        skytrax_labels (bool, optional): If True, use Skytrax rating classification. Defaults to True.
        real_labels (DataFrame, optional): Real aspect labels for reviews. Needed if skytrax_labels=False.
        allowed_keys (list, optional): Aspects allowed for extraction. Defaults to common Skytrax aspects.

    Returns:
        tuple:
            - pred_sentences_link_reviews (dict): Mapping of sentences linked to review indices by aspect.
            - AE_pred_label (dict): Predicted labels for extracted aspect sentences.
    """
    data_train_copy = data_train.copy()

    def filter_dict_by_keys(input_dict, allowed_keys):
        """Filter a dictionary by allowed keys."""
        return {key: value for key, value in input_dict.items() if key in allowed_keys}

    aspect_keywords = filter_dict_by_keys(aspect_keywords, allowed_keys)

    if skytrax_labels is True:
        def classify(x):
            """Classify Skytrax ratings into binary labels (0: Negative, 1: Positive)."""
            if x in [1, 2, 3]:
                return 0  # NEG
            elif x in [4, 5]:
                return 1  # POS
            return x

        data_train_copy[allowed_keys] = data_train_copy[allowed_keys].applymap(classify)

    elif skytrax_labels is False:
        def classify(x):
            """Classify real labels into binary labels (0: Negative, 1: Positive)."""
            if x in ['NEG', 'NEU']:
                return 0  # NEG
            elif x == 'POS':
                return 1  # POS
            return x

        real_labels_copy = real_labels.copy()
        real_labels_copy[allowed_keys] = real_labels_copy[allowed_keys].applymap(classify)

    else:
        raise TypeError('skytrax_labels should be True or False')

    def find_aspect_indices(df, aspect_keywords):
        """Find sentences that match aspect keywords in the data.

        Args:
            df (DataFrame): Data containing reviews.
            aspect_keywords (dict): Dictionary of aspect keywords.

        Returns:
            tuple:
                - aspect_sentences (dict): Sentences matching keywords per aspect.
                - dict_reviews (dict): Dictionary linking sentences to review indices.
        """
        aspect_sentences = {}
        dict_reviews, list_sentences = split_and_join_sentences(df, n=1)
        sentences_df = pd.DataFrame(list_sentences, columns=['Sentence'])

        print('started')
        sentences_df['Lemmatized_Sentence'] = lemmatize_sentences(sentences_df['Sentence'].tolist())
        print('ended')

        for aspect, keywords in aspect_keywords.items():
            if keywords:
                pattern = '|'.join(keywords)
                sentences = sentences_df[sentences_df['Lemmatized_Sentence'].str.contains(pattern, case=False)][
                    'Sentence'].tolist()
            else:
                sentences = []
            aspect_sentences[aspect] = sentences

        return aspect_sentences, dict_reviews

    aspect_sentences, dict_reviews = find_aspect_indices(data_train_copy, aspect_keywords)

    reverse_index_dict = {value: index for index, values in dict_reviews.items() for value in values}

    pred_sentences_link_reviews = {}
    AE_pred_label = {}

    for aspect in aspect_keywords.keys():
        pred_sentences_link_reviews[aspect] = {}
        AE_pred_label[aspect] = {}
        for sentence in aspect_sentences[aspect]:
            index = reverse_index_dict[sentence]
            pred_sentences_link_reviews[aspect][sentence] = index

            if skytrax_labels is True:
                AE_pred_label[aspect][sentence] = data_train_copy[data_train_copy['Index'] == index][aspect].values[0]
            else:
                AE_pred_label[aspect][sentence] = real_labels_copy[real_labels_copy['Index'] == index][aspect].values[0]

    return pred_sentences_link_reviews, AE_pred_label


class TextDataset(Dataset):
    """Custom dataset for handling training sentences and label pairs with tokenization.

    Args:
        texts (list): List of input training sentences.
        labels (list): List of labels corresponding to the sentences.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert text to tokens.
        max_len (int): Maximum length for tokenized sequences.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Retrieves the tokenized representation of a sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and label tensors.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # Ensures each sequence is padded to max_length
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }



def collate_fn(batch):
    """Collates a batch of samples into padded tensors for input and attention masks.
    Custom collate_fn function to use as arg in DataLoader from torch.utils.data (used in compute_prediction function for instance)

    Args:
        batch (list): A list of dictionaries with 'input_ids', 'attention_mask', and 'label' tensors.

    Returns:
        dict: A dictionary containing padded input_ids, attention_mask, and label tensors.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': labels
    }


class BertForSequenceClassificationWithMoreLayers(BertPreTrainedModel):
    """BERT model with additional fully connected layers for sequence classification.

    Args:
        config (BertConfig): Configuration object for the BERT model.
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Define additional layers
        self.additional_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation1 = nn.ReLU()
        self.additional_layer2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.activation2 = nn.ReLU()
        self.additional_layer3 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.activation3 = nn.ReLU()

        self.classifier = nn.Linear(config.hidden_size // 4, config.num_labels)

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """Performs a forward pass through the BERT model with additional layers.

        Args:
            input_ids (torch.Tensor, optional): Input tensor with token IDs.
            attention_mask (torch.Tensor, optional): Mask to avoid attention on padding tokens.
            token_type_ids (torch.Tensor, optional): Segment token IDs.
            position_ids (torch.Tensor, optional): Positional embeddings.
            labels (torch.Tensor, optional): Labels for computing the loss.
            return_dict (bool, optional): If True, returns a `SequenceClassifierOutput` object.

        Returns:
            Union[Tuple[torch.Tensor], SequenceClassifierOutput]: Model outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[1]  # Pooled output
        sequence_output = self.dropout(sequence_output)

        # Sequentially pass through additional layers
        sequence_output = self.additional_layer1(sequence_output)
        sequence_output = self.activation1(sequence_output)

        sequence_output = self.additional_layer2(sequence_output)
        sequence_output = self.activation2(sequence_output)

        sequence_output = self.additional_layer3(sequence_output)
        sequence_output = self.activation3(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def eval_model(model, dataloader, device):
    """
    Evaluate the performance of a deep learning ASC supervised model on a given dataloader.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the evaluation dataset.
            - accuracy (float): The accuracy of the model on the evaluation dataset.
            - precision (float): The precision score of the model.
            - recall (float): The recall score of the model.
            - f1 (float): The F1 score of the model.
    """

    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # No gradients needed for evaluation, reduces memory usage and speeds up computation
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += torch.sum(predictions == labels).item()

            # Collect all predictions and labels for metric calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    return avg_loss, accuracy, precision, recall, f1


def train_and_eval_epoch(model, train_dataloader, val_dataloader, optimizer, device, scheduler, global_step, eval_step,
                         save_best_model_path, best_val_score=None):
    """
    Train and evaluate the model for one epoch.

    This function trains the model on the training dataset and evaluates it on the validation dataset at specified intervals.
    It tracks training and validation loss, accuracy, precision, recall, and F1 score, saving the best model based on
    validation F1 score.


    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to perform computations on (CPU or GPU).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        global_step (int): Current global step count, to track training progress.
        eval_step (int): Step interval for evaluation.
        save_best_model_path (str): File path to save the best model.
        best_val_score (float, optional): Current best validation score. Defaults to None.

    Returns:
        dict: A dictionary containing training and validation metrics for the epoch.
        int: Updated global step count after training and evaluation.
        float: The best validation score achieved during training.
    """

    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    step_results = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }
    if best_val_score == None:
        best_val_score = float('-inf')  # Initialize with the lowest possible value
    # Determine total steps in the current epoch to identify the last batch
    total_batches = len(train_dataloader)
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        _, predictions = torch.max(outputs.logits, dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_examples += labels.size(0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1  # Increment the global step count

        # Evaluate based on the global step count or ensure evaluation at the last batch of the epoch
        if global_step % eval_step == 0 or (i + 1) == total_batches:
            avg_train_loss = total_loss / total_batches
            train_accuracy = total_correct / total_examples
            avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = eval_model(model, val_dataloader, device)

            step_results['train_loss'].append(avg_train_loss)
            step_results['train_accuracy'].append(train_accuracy)
            step_results['val_loss'].append(avg_val_loss)
            step_results['val_accuracy'].append(val_accuracy)
            step_results['val_precision'].append(val_precision)
            step_results['val_recall'].append(val_recall)
            step_results['val_f1_score'].append(val_f1)

            # Check if the current validation score is the best
            current_val_score = step_results['val_f1_score'][-1]  # Assuming validation accuracy as the score
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                torch.save(model.state_dict(), save_best_model_path)
                print(f'Saved new best model with validation f1 score: {best_val_score * 100:.2f}%')

            # Optionally reset metrics after each evaluation to start fresh
            total_loss = 0
            total_correct = 0
            total_examples = 0

    return step_results, global_step, best_val_score


def get_AE_prediction_dict(data_test, aspect, aspect_keywords, skytrax_labels=True, real_labels=None):
    """
    get_AE_prediction_dict is a higher-level function that prepares and filters the data before passing it to aspect_extraction_keywords
    Performs aspect extraction (LKD) predictions from the test dataset.

    This function filters the test dataset based on the provided aspect and aspect_keywords dict,
    and then performs aspect extraction using keywords to link sentences to their respective reviews.

    Args:
        data_test (pd.DataFrame): The dataset containing review data to be tested.
        aspect (str): The aspect to be extracted from the reviews.
        aspect_keywords (dict): A dictionary mapping aspects to their corresponding keywords for extraction.
        skytrax_labels (bool, optional): If True, uses Skytrax ratings for filtering. Defaults to True.
        real_labels (pd.DataFrame, optional): The dataset containing true sentiment labels, required if skytrax_labels is False.

    Returns:
        tuple: A tuple containing:
            - pred_sentences_link_reviews (dict): A dictionary mapping extracted sentences to their respective review indices.
            - AE_pred_label (dict): A dictionary mapping extracted sentences to their predicted sentiment labels.
    """

    # Testing data: full_test_set
    if skytrax_labels == False:
        # data_test = data[data['Index'].isin(full_test_set_2['Index'])]
        real_labels_nans_index = real_labels[real_labels[aspect].isna()]['Index']
        data_test = data_test[~data_test['Index'].isin(real_labels_nans_index)]

    if skytrax_labels == True:
        # data_test = data[data['Index'].isin(full_test_set_2['Index'])]
        data_test.dropna(inplace=True, subset=aspect)

    allowed_keys = [aspect]
    # allowed_keys = ['seat comfort', 'crew flight','food and beverages', 'on ground services', 'entertainment']
    pred_sentences_link_reviews, AE_pred_label = aspect_extraction_keywords(aspect_keywords, data_test,
                                                                            skytrax_labels=skytrax_labels,
                                                                            allowed_keys=allowed_keys,
                                                                            real_labels=real_labels)
    return pred_sentences_link_reviews, AE_pred_label


def compute_prediction(model_type, state_dict_path, aspect, AE_pred_label, probability=False):
    """Compute predictions for a given aspect using a pre-defined supervised model that is already trained.

    Args:
        model_type (nn.Module): Model class to be used for predictions.
        state_dict_path (str): Path to the model's state dictionary.
        aspect (str): Aspect for which predictions are being made.
        AE_pred_label (dict): Dictionary containing sentences and their corresponding labels. From aspect_extraction_keywords.
        probability (bool, optional): If True, returns probabilities for class '1'. Defaults to False.

    Returns:
        dict: A dictionary mapping sentences to predicted labels or probabilities.
    """
    # Initialize the model configuration and model instance
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = model_type(config)

    # Load the model state dictionary
    model_state_dict = torch.load(state_dict_path)
    model.load_state_dict(model_state_dict)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare testing sentences and labels
    testing_sentences = list(AE_pred_label[aspect].keys())
    testing_labels = list(AE_pred_label[aspect].values())

    # Create dataset and dataloader
    max_len = 128
    dataset = TextDataset(testing_sentences, testing_labels, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    predicted_labels = []
    true_labels = []
    probabilities = [] if probability else None

    model.eval()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)

            if probability:
                probs_class_1 = probs[:, 1]  # Extract the probability of class '1'
                probabilities.extend(probs_class_1.cpu().numpy().tolist())
                print(f'Predictions: {probs_class_1}, Labels: {labels}')
            else:
                print(f'Predictions: {predictions}, Labels: {labels}')

        predicted_labels.extend(predictions.cpu().numpy().tolist())
        true_labels.extend(labels.cpu().numpy().tolist())

    # Prepare result dictionary
    dict_result = {}
    for i, sentence in enumerate(testing_sentences):
        if probability:
            dict_result[sentence] = probabilities[i]
        else:
            dict_result[sentence] = predicted_labels[i]

    print('length predictions:', len(dict_result.keys()))

    return dict_result



def evaluate_sentiment_classification_nn2(data_test, aspect, dict_result, pred_sentences_link_reviews,
                                          skytrax_labels=True, real_labels=None, probability=False, threshold=0.5):
    """Evaluate sentiment classification using predictions and true labels. Uses output of compute_prediction (dict_result).
    Plots confusion matrices

    Args:
        data_test (DataFrame): Test dataset containing the true labels.
        aspect (str): Aspect for which predictions are being evaluated.
        dict_result (dict): Dictionary of sentences and their predicted values.
        pred_sentences_link_reviews (dict): Links sentences to their review index.
        skytrax_labels (bool, optional): If True, use `data_test` for true labels. Defaults to True.
        real_labels (DataFrame, optional): Real labels if `skytrax_labels` is False. Defaults to None.
        probability (bool, optional): If True, return probability values. Defaults to False.
        threshold (float, optional): Probability threshold for classification. Defaults to 0.5.

    Returns:
        float: Accuracy of the sentiment classification.
        DataFrame (optional): DataFrame with calculated scores if `probability` is True.
    """
    # Get real labels
    if skytrax_labels:
        real_labels = data_test

    # Predictions and sentences
    predicted_labels_vader = list(dict_result.values())
    sentences_vader = list(dict_result.keys())

    # Link predictions to review index and group by index
    test_link_dict = {}
    for enum, prediction in enumerate(predicted_labels_vader):
        index = pred_sentences_link_reviews[aspect][sentences_vader[enum]]
        test_link_dict.setdefault(index, []).append(prediction)

    # Filter out indices with NaN labels
    test_link_dict = {key: value for key, value in test_link_dict.items() if not pd.isna(real_labels[real_labels['Index'] == key][aspect].values[0])}

    # Average predictions for each index
    test_final_pred_dict = {key: np.mean(value) for key, value in test_link_dict.items()}
    test_final_pred_df = pd.DataFrame.from_dict(test_final_pred_dict, orient='index', columns=['proba_1']).reset_index()
    test_final_pred_df.rename(columns={'index': 'Index'}, inplace=True)
    test_final_pred_df['pred'] = np.where(test_final_pred_df['proba_1'] <= threshold, 0, 1)

    # Get true labels
    list_indices = list(test_link_dict.keys())
    if skytrax_labels:
        df_true_labels = data_test[data_test['Index'].isin(list_indices)][['Index', aspect]].dropna()
        df_true_labels[aspect] = np.where(df_true_labels[aspect] <= 3, 0, 1)
    else:
        df_true_labels = real_labels[real_labels['Index'].isin(list_indices)][['Index', aspect]]
        df_true_labels[aspect] = np.where(df_true_labels[aspect] == 'POS', 1, 0)

    # Merge predictions with true labels and calculate metrics
    df_calc_score = df_true_labels.merge(test_final_pred_df, on='Index')
    accuracy = (df_calc_score[aspect] == df_calc_score['pred']).mean()
    F1_score = f1_score(df_calc_score[aspect], df_calc_score['pred'])

    # Precision, recall, and F1 scores for each class
    precision, recall, f1, support = precision_recall_fscore_support(df_calc_score[aspect], df_calc_score['pred'], average=None, labels=[0, 1])
    _, _, f1_macro, _ = precision_recall_fscore_support(df_calc_score[aspect], df_calc_score['pred'], average='macro', labels=[0, 1])

    # Display results
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score: {F1_score:.3f}')
    print(f'Precision (Negative): {precision[0]:.3f}')
    print(f'Recall (Negative): {recall[0]:.3f}')
    print(f'F1 Score (Negative): {f1[0]:.3f}')
    print(f'Precision (Positive): {precision[1]:.3f}')
    print(f'Recall (Positive): {recall[1]:.3f}')
    print(f'F1 Score (Positive): {f1[1]:.3f}')
    print(f'F1 Score (Macro): {f1_macro:.3f}')

    # Display confusion matrices
    ConfusionMatrixDisplay.from_predictions(df_calc_score[aspect], df_calc_score['pred'], normalize='true', cmap='Blues')
    plt.title(f'{aspect}, normalized on true labels')
    plt.show()

    ConfusionMatrixDisplay.from_predictions(df_calc_score[aspect], df_calc_score['pred'], normalize='pred', cmap='Reds')
    plt.title(f'{aspect}, normalized on predicted labels')
    plt.show()

    if probability:
        return accuracy, df_calc_score
    return accuracy


class SelfAttention(nn.Module):
    """Self-attention mechanism for processing sequences.

    This class implements a multi-head self-attention layer with dropout and layer normalization.
    Used in BertForSequenceClassificationWithSelfAttention

    Args:
        embed_dim (int): Dimension of the input embeddings.
        num_heads (int): Number of attention heads.

    Attributes:
        multihead_attn (MultiheadAttention): Multi-head attention layer.
        dropout (Dropout): Dropout layer for regularization.
        norm (LayerNorm): Layer normalization for stabilizing training.
    """

    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads)
        self.dropout = Dropout(0.2)
        self.norm = LayerNorm(embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        """Forward pass through the self-attention layer.

        Args:
            query (Tensor): The query tensor of shape (batch_size, seq_length, embed_dim).
            key (Tensor): The key tensor of shape (batch_size, seq_length, embed_dim).
            value (Tensor): The value tensor of shape (batch_size, seq_length, embed_dim).
            attn_mask (Tensor, optional): Attention mask for masking out certain positions.

        Returns:
            Tensor: The output tensor after applying self-attention, dropout, and layer normalization.
        """
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)  # Apply residual connection followed by layer normalization


class BertForSequenceClassificationWithSelfAttention(BertPreTrainedModel):
    """BERT model for sequence classification with self-attention enhancements.

    This class extends BERT with additional layers and self-attention mechanisms for improved sequence classification tasks.

    Args:
        config (BertConfig): Configuration object for the BERT model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # Define additional layers
        self.additional_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation1 = nn.ReLU()

        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(config.hidden_size, num_heads=4) for _ in range(4)
        ])

        # Define subsequent linear layers
        self.additional_layer2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(classifier_dropout)
        self.additional_layer3 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(classifier_dropout)
        self.additional_layer4 = nn.Linear(config.hidden_size // 4, config.hidden_size // 8)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size // 8, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """Forward pass through the model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
            token_type_ids (torch.Tensor, optional): Segment IDs for distinguishing sentences.
            position_ids (torch.Tensor, optional): Position IDs for tokens.
            head_mask (torch.Tensor, optional): Mask for attention heads.
            inputs_embeds (torch.Tensor, optional): Input embeddings instead of input IDs.
            labels (torch.Tensor, optional): Labels for computing loss.
            output_attentions (bool, optional): Whether to return attention weights.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dict or tuple.

        Returns:
            Union[Tuple[torch.Tensor], SequenceClassifierOutput]: The logits and optionally the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[1]  # Pooled output
        sequence_output = self.dropout(sequence_output)

        # Pass through first additional layer
        sequence_output = self.additional_layer1(sequence_output)
        sequence_output = self.activation1(sequence_output)

        # Self-attention mechanism
        for self_attention in self.self_attention_layers:
            sequence_output = sequence_output.unsqueeze(0)  # Adjust dimensions for attention layer
            sequence_output = self_attention(sequence_output, sequence_output, sequence_output)
            sequence_output = sequence_output.squeeze(0)  # Adjust dimensions back

        # Additional processing
        sequence_output = self.additional_layer2(sequence_output)
        sequence_output = self.activation2(sequence_output)
        sequence_output = self.dropout2(sequence_output)
        sequence_output = self.additional_layer3(sequence_output)
        sequence_output = self.activation3(sequence_output)
        sequence_output = self.dropout3(sequence_output)
        sequence_output = self.additional_layer4(sequence_output)
        sequence_output = self.activation4(sequence_output)
        sequence_output = self.dropout4(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, logits, labels):
        """Computes the loss based on the given logits and labels.

        Args:
            logits (torch.Tensor): The logits from the model.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return loss



