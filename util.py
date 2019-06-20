
"""
***************************************************************************************
*
*                   Yara English to Cypher Neural Machine Translation
*
*
*  Name : Idaly Ali
*
*  Designation : Research Scientist
*
*  Description : Seq2Seq: Neural Machine Translation with Attention for translation
*                 from English to Cypher - Utilities
*
*  Reference: TensorFlow 2.0-Alpha Tutorials
*
***************************************************************************************

"""


########################

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import unicodedata
import re

########################

#
# def unicode_to_ascii(s):
#     """
#     Takes in a unicode string, outputs ASCII equivalent
#     :param s: String of unicode
#     :return: String of ASCII
#     """
#     return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
#
#
# def preprocess_sentence_english(w):
#     """
#     Converts english question to inputs for NMT
#     :param w: String of Enqlish question
#     :return w: String compatible with NMT
#     """
#     w = unicode_to_ascii(w.lower().strip())
#
#     # Creates spaces between a word and the following punctuations
#     w = re.sub(r"([?.!,¿])", r" \1 ", w)
#     w = re.sub(r'[" "]+', " ", w)
#
#     # Replace every characters except (a-z, A-Z, ".", "?", "!", ",")
#     w = re.sub(r"[^a-zA-Z?.!,¿{}[]():->]+", " ", w)
#     w = w.rstrip().strip()
#
#     # Append start and end tags to each either ends of sentence so that the NMT model knows when sentences begin
#     w = '<start> ' + w + ' <end>'
#
#     return w
#
#
# def preprocess_sentence_cypher(w):
#     """
#     Converts Cypher answers to inputs for NMT
#     :param w: String of Enqlish question
#     :return w: String compatible with NMT
#     """
#     # Converts to lowercase
#     w = unicode_to_ascii(w.lower().strip())
#
#     # Creates spaces between a word and the following punctuations
#     w = re.sub(r"([?;!,¿])", r" \1 ", w)
#     w = re.sub(r'[" "]+', " ", w)
#
#     # Replace every characters except (a-z, A-Z, ".", "?", "!", ",")
#     w = re.sub(r"[^a-zA-Z?.!,¿{}[]():->]+", " ", w)
#     w = w.rstrip().strip()
#
#     # Append start and end tags to each either ends of sentence so that the NMT model knows when sentences begin
#     w = '<start> ' + w + ' <end>'
#
#     return w

def unicode_to_ascii(s):
    """
    Takes in a unicode string, outputs ASCII equivalent
    :param s: String of unicode
    :return: String of ASCII
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence_english(w):
    """
    Converts english question to inputs for NMT
    :param w: String of Enqlish question
    :return w: String compatible with NMT
    """
    w = unicode_to_ascii(w.lower().strip())

    # create space between a word and the following punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿{}[]():->]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to each sentence so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def preprocess_sentence_cypher(w):
    """
    Converts Cypher answers to inputs for NMT
    :param w: String of Enqlish question
    :return w: String compatible with NMT
    """
    w = unicode_to_ascii(w.lower().strip())

    # create space between a word and the following punctuation
    w = re.sub(r"([?;!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿{}[]():->]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to each sentence so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH):
    """
    Create dataset using texts of English questions and Cypher answers
    :param ENGLISH_TXT_PATH: Path to list of English questions
    :param CYPHER_TXT_PATH: Path to list of Cypher answers
    :return: List of Cypher and English datasets
    """
    english = []
    cypher = []

    with open(ENGLISH_TXT_PATH) as infile:
        for line in infile:
            if line:
                processed_line = preprocess_sentence_english(line)
                english.append(processed_line)

    with open(CYPHER_TXT_PATH) as infile:
        for line in infile:
            if line:
                processed_line = preprocess_sentence_cypher(line)
                cypher.append(processed_line)

    return cypher, english


def max_length(tensor):
    """
    Returns maximum length of input tensors
    :param tensor: Input tensors
    :return: Maximum length of input tensors
    """
    return max(len(t) for t in tensor)


# In[12]:


def tokenize(lang):
    """
    Tokenize and return sentences, along with tensor of sequences
    :param lang: Sentences
    :return: Tensor of sequences, Tokenizer object
    """

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# In[13]:


def load_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH):
    """
    Load dataset
    :param ENGLISH_TXT_PATH: Path to list of English questions
    :param CYPHER_TXT_PATH: Path to list of Cypher answers
    :return:
    """

    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# def create_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH):
#     """
#     Create dataset using texts of English questions and Cypher answers
#     :param ENGLISH_TXT_PATH: Path to list of English questions
#     :param CYPHER_TXT_PATH: Path to list of Cypher answers
#     :return: List of Cypher and English datasets
#     """
#
#     english = []
#     cypher = []
#
#     with open(ENGLISH_TXT_PATH) as infile:
#         for line in infile:
#             if line:
#                 processed_line = preprocess_sentence_english(line)
#                 english.append(processed_line)
#
#     with open(CYPHER_TXT_PATH) as infile:
#         for line in infile:
#             if line:
#                 processed_line = preprocess_sentence_cypher(line)
#                 cypher.append(processed_line)
#
#     return cypher, english
#
#
# def max_length(tensor):
#     """
#     Returns maximum length of input tensors
#     :param tensor: Input tensors
#     :return: Maximum length of input tensors
#     """
#     return max(len(t) for t in tensor)
#
#
# def tokenize(lang):
#     """
#     Tokenize and return sentences, along with tensor of sequences
#     :param lang: Sentences
#     :return: Tensor of sequences, Tokenizer object
#     """
#
#     lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
#     lang_tokenizer.fit_on_texts(lang)
#     tensor = lang_tokenizer.texts_to_sequences(lang)
#     tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
#     return tensor, lang_tokenizer


# def load_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH):
#     """
#     Load dataset
#     :param ENGLISH_TXT_PATH: Path to list of English questions
#     :param CYPHER_TXT_PATH: Path to list of Cypher answers
#     :return:
#     """
#
#     # Create cleaned input and output pairs
#     targ_lang, inp_lang = create_dataset(ENGLISH_TXT_PATH, CYPHER_TXT_PATH)
#
#     input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
#     target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
#
#     return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
#
# def convert(lang, tensor):
#     """
#     Prints mapping of sentence to tenors
#     :param lang: Tokenized inputs
#     :param tensor: Input tensors
#     """
#
#     for t in tensor:
#         if t != 0:
#             print('%d -----> %s' % (t, lang.index_word[t]))
#