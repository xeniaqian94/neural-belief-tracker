# -*- coding: utf-8 -*-
# !/usr/local/bin/python
import codecs
import math

import numpy

'''
This script supports all dictionary IO, helper function, etc. 
'''


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word.

    We hash the word to always get the same vector for the given word, e.g. slot or value
    """

    seed_value = hash_string(word)
    numpy.random.seed(seed_value)

    neg_value = - math.sqrt(6) / math.sqrt(D)
    pos_value = math.sqrt(6) / math.sqrt(D)

    rsample = numpy.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = numpy.linalg.norm(rsample)
    rsample_normed = rsample / norm

    return rsample_normed


def hash_string(s):
    return abs(hash(s)) % (10 ** 8)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = numpy.exp(x)
    sf = sf / numpy.sum(sf, axis=0)
    return sf


def load_word_vectors(file_destination, primary_language="english"):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    # print "XAVIER - returning null dictionary"
    # return {}

    print("Loading pretrained word vectors from", file_destination, "- treating", primary_language,
          "as the primary language.")
    word_dictionary = {}

    lp = {}
    lp["english"] = u"en_"
    lp["german"] = u"de_"
    lp["italian"] = u"it_"
    lp["russian"] = u"ru_"
    lp["sh"] = u"sh_"
    lp["bulgarian"] = u"bg_"
    lp["polish"] = u"pl_"
    lp["spanish"] = u"es_"
    lp["french"] = u"fr_"
    lp["portuguese"] = u"pt_"
    lp["swedish"] = u"sv_"
    lp["dutch"] = u"nl_"

    language_key = lp[primary_language]

    f = codecs.open(file_destination, 'r', 'utf-8')

    for line in f:

        line = line.split(" ", 1)
        transformed_key = line[0].lower()

        if language_key in transformed_key:  # only handles the prefix_paraphrase_semantic_embedding

            transformed_key = transformed_key.replace(language_key, "")

            try:
                transformed_key = str(transformed_key)
            except:
                # print("Can't convert the key to unicode:", transformed_key)
                continue

            word_dictionary[transformed_key] = numpy.fromstring(line[1], dtype="float32", sep=" ")

            if word_dictionary[transformed_key].shape[0] != 300:
                print(transformed_key, word_dictionary[transformed_key].shape)

    print(len(word_dictionary), "vectors loaded from",
          file_destination)  # it could take a few minutes to load 86407 words in english

    return normalise_word_vectors(word_dictionary)


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def w2i(key_list):
    return {k: v for v, k in enumerate(key_list)}


def i2w(key_list):
    assert len(key_list) == len(set(key_list))
    return {v: k for v, k in enumerate(key_list)}
