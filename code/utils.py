# -*- coding: utf-8 -*-
# !/usr/local/bin/python
import codecs
import json
import math
import random
import string
from copy import deepcopy

import numpy as np
import torch
from torch import nn

'''
This script supports all dictionary IO, helper function, etc. 
'''


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word.

    We hash the word to always get the same vector for the given word, e.g. slot or value
    """

    seed_value = hash_string(word)
    np.random.seed(seed_value)

    neg_value = - math.sqrt(6) / math.sqrt(D)
    pos_value = math.sqrt(6) / math.sqrt(D)

    rsample = np.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = np.linalg.norm(rsample)
    rsample_normed = rsample / norm

    return rsample_normed


def hash_string(s):
    return abs(hash(s)) % (10 ** 8)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf / np.sum(sf, axis=0)
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
    word_dictionary["<eos>"] = xavier_vector("<eos>")

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

            word_dictionary[transformed_key] = np.fromstring(line[1], dtype="float32", sep=" ")

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


def process_turn_hyp(transcription, language):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language.
    """
    exclude = set(string.punctuation)
    exclude.remove("'")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    if language == "it" or language == "italian":  # or language == "en" or language == "english":
        transcription = transcription.replace("'", " ")
    if language == "en" or language == "english":  # or language == "en" or language == "english":
        transcription = transcription.replace("'", "")

    return transcription


def process_woz_dialogue(woz_dialogue, language, override_en_ontology):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en" or override_en_ontology:
        null_bs = {}
        null_bs["food"] = "none"
        null_bs["price range"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["food", "price range", "area"]
        pure_requestables = ["address", "phone", "postcode"]  # what happened to name

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["area"] = "none"
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["gegend"] = "none"
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    # user talks first, so there is no requested DA initially.
    current_req = [""]

    current_conf_slot = [""]
    current_conf_value = [""]

    lp = {}
    lp["german"] = u"de_"
    lp["italian"] = u"it_"

    for idx, turn in enumerate(woz_dialogue):

        current_DA = turn["system_acts"]

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if each_da in informable_slots:
                current_req.append(each_da)
            elif each_da in pure_requestables:
                current_conf_slot.append("request")
                current_conf_value.append(each_da)
            else:
                if type(each_da) is list:   # e.g. [["area","dontcare"]]
                    current_conf_slot.append(each_da[0])
                    current_conf_value.append(each_da[1])

        if not current_req:
            current_req = [""]

        if not current_conf_slot:
            current_conf_slot = [""]
            current_conf_value = [""]

        current_transcription = turn["transcript"]
        current_transcription = process_turn_hyp(current_transcription, language)

        read_asr = turn["asr"]

        current_asr = []

        for (hyp, score) in read_asr:
            current_hyp = process_turn_hyp(hyp, language)
            current_asr.append((current_hyp, score))

        old_trans = current_transcription

        exclude = set(string.punctuation)
        exclude.remove("'")

        current_transcription = ''.join(ch for ch in current_transcription if ch not in exclude)
        current_transcription = current_transcription.lower()

        current_labels = turn["turn_label"]

        current_bs = deepcopy(prev_belief_state)

        # print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []  # reset requestables at each turn

        for label in current_labels:
            (c_slot, c_value) = label

            if c_slot in informable_slots:
                current_bs[c_slot] = c_value

            elif c_slot == "request":
                current_bs["request"].append(c_value)

        curr_lab_dict = {}
        for x in current_labels:
            if x[0] != "request":
                curr_lab_dict[x[0]] = x[1]

        dialogue_representation.append(((current_transcription, current_asr), current_req, current_conf_slot,
                                        current_conf_value, deepcopy(current_bs), deepcopy(prev_belief_state)))

        prev_belief_state = deepcopy(current_bs)

    return dialogue_representation


def load_woz_data(file_path, language, percentage=1.0, override_en_ontology=False):
    """
    This method loads WOZ dataset as a collection of utterances.
    Testing means load everything, no split.
    Return: a tuple of pairs. Each pair has strctured utterance+label, with unstrctured JSON object
    """

    woz_json = json.load(codecs.open(file_path, "r", "utf-8"))
    dialogues = []

    training_turns = []

    dialogue_count = len(woz_json)

    percentage = float(percentage)
    dialogue_count = int(percentage * float(dialogue_count))

    if dialogue_count != 200:
        print("Percentage is:", percentage, "so loading:", dialogue_count)

    for idx in range(0, dialogue_count):

        current_dialogue = process_woz_dialogue(woz_json[idx]["dialogue"], language, override_en_ontology)
        dialogues.append(current_dialogue)

        for turn_idx, turn in enumerate(current_dialogue):

            # if turn_idx == 0:
            #     prev_turn_label = []
            # else:
            #     prev_turn_label = current_label

            current_label = []

            for req_slot in turn[4]["request"]:
                current_label.append(("request", req_slot))
                # print "adding reqslot:", req_slot

            # this now includes requests:
            for inf_slot in turn[4]:
                # print (inf_slot, turn[5][inf_slot])
                if inf_slot != "request":
                    current_label.append((inf_slot, turn[4][inf_slot]))
            #                    if inf_slot == "request":
            #                        print "!!!!!", inf_slot, turn[5][inf_slot]

            current_utterance = (turn[0], turn[1], turn[2], turn[3], current_label,
                                 turn[5])  # turn [5] is the past belief state

            # print "$$$$", current_utterance

            training_turns.append(current_utterance)

    # print "Number of utterances in", file_path, "is:", len(training_turns)

    return (dialogues, training_turns)


def binary_mask(example, requestable_count):
    """
    takes a list, i.e. 2,3,4, and if req count is 8, returns: 00111000
    """

    zeros = torch.Tensor(np.zeros((requestable_count,), dtype=np.float32))
    for x in example:
        zeros[x] = 1

    return zeros


def delexicalise_utterance_values(utterance, target_slot, target_values):
    """
    Takes a list of words which represent the current utterance, the loaded vectors, finds all occurrences of both slot name and slot value,
    and then returns the updated vector with "delexicalised tag" in them.
    """

    if type(utterance) is list:
        utterance = " ".join(utterance)

    if target_slot == "request":
        value_count = len(target_values)
    else:
        value_count = len(target_values) + 1

    delexicalised_vector = torch.FloatTensor(np.zeros((value_count,),dtype="float32"))

    for idx, target_value in enumerate(target_values):
        if " " + target_value + " " in utterance:
            delexicalised_vector[idx] = 1.0

    return delexicalised_vector
