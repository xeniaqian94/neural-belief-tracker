# -*- coding: utf-8 -*-
# !/usr/local/bin/python
import ConfigParser
import codecs
import json
import os
import random
import time

import numpy
# import tensorflow as tf

from code.models_py import model_definition
from code.utils import load_word_vectors, xavier_vector


class NeuralBeliefTracker:
    """
    Call to initialise the model with pre-trained parameters and given ontology.
    """

    def __init__(self, config_filepath):

        config = ConfigParser.RawConfigParser()
        self.config = config

        try:
            config.read(config_filepath)
        except:
            print("Couldn't read config file from", config_filepath, "... aborting.")
            return None

        dataset_name = config.get("model", "dataset_name")

        word_vectors = {}
        word_vector_destination = config.get("data", "word_vectors")

        lp = {}
        lp["english"] = u"en"
        lp["german"] = u"de"
        lp["italian"] = u"it"

        try:
            language = config.get("model", "language")
            language_suffix = lp[language]
        except:
            language = "english"
            language_suffix = lp[language]

        self.language = language
        self.language_suffix = language_suffix

        self.num_models = int(config.get("model", "num_models"))

        self.batches_per_epoch = int(config.get("train", "batches_per_epoch"))
        self.max_epoch = int(config.get("train", "max_epoch"))
        self.batch_size = int(config.get("train", "batch_size"))

        if not os.path.isfile(word_vector_destination):
            print("Vectors not there, downloading small Paragram and putting it there.")
            os.system("wget https://mi.eng.cam.ac.uk/~nm480/prefix_paragram.txt")
            os.system("mkdir -p word-vectors/")  # create a directory and any parents that don't already exist.
            os.system("mkdir -p models/")
            os.system("mkdir -p results/")
            os.system("mv prefix_paragram.txt " + word_vector_destination)

        word_vectors = load_word_vectors(word_vector_destination, primary_language=language)  # dict {} type

        word_vectors["tag-slot"] = xavier_vector("tag-slot")
        word_vectors["tag-value"] = xavier_vector("tag-value")

        ontology_filepath = config.get("model", "ontology_filepath")
        dialogue_ontology = json.load(codecs.open(ontology_filepath, "r", "utf-8"))
        dialogue_ontology = dialogue_ontology["informable"]
        slots = dialogue_ontology.keys()  # ["address","price range",...]

        word_vector_size = random.choice(word_vectors.values()).shape[0]

        # a bit of hard-coding to make our lives easier.
        if u"price" in word_vectors and u"range" in word_vectors:
            word_vectors[u"price range"] = word_vectors[u"price"] + word_vectors[u"range"]
        if u"post" in word_vectors and u"code" in word_vectors:
            word_vectors[u"postcode"] = word_vectors[u"post"] + word_vectors[u"code"]
        if u"dont" in word_vectors and u"care" in word_vectors:
            word_vectors[u"dontcare"] = word_vectors[u"dont"] + word_vectors[u"care"]
        if u"addressess" in word_vectors:
            word_vectors[u"addressess"] = word_vectors[u"addresses"]
        if u"dont" in word_vectors:
            word_vectors[u"don't"] = word_vectors[u"dont"]

        if language == "italian":
            word_vectors["dontcare"] = word_vectors["non"] + word_vectors["importa"]
            word_vectors["non importa"] = word_vectors["non"] + word_vectors["importa"]

        if language == "german":
            word_vectors["dontcare"] = word_vectors["es"] + word_vectors["ist"] + word_vectors["egal"]
            word_vectors["es ist egal"] = word_vectors["es"] + word_vectors["ist"] + word_vectors["egal"]

        exp_name = config.get("data", "exp_name")  # what does this mean

        config_model_type = config.get("model", "model_type")
        use_cnn = False

        if config_model_type == "cnn":
            print("----------- Config Model Type:", config_model_type, "-------------")
            use_cnn = True
            model_type = "CNN"
        elif config_model_type == "dnn":
            print("----------- Config Model Type:", config_model_type, "-------------")
            model_type = "DNN"

        self.value_specific_decoder = config.get("model", "value_specific_decoder")

        if self.value_specific_decoder in ["True", "true"]:
            self.value_specific_decoder = True
        else:
            self.value_specific_decoder = False

        self.learn_belief_state_update = config.get("model", "learn_belief_state_update")

        if self.learn_belief_state_update in ["True", "true"]:
            self.learn_belief_state_update = True
        else:
            self.learn_belief_state_update = False

        print("value_specific_decoder", self.value_specific_decoder)
        print("learn_belief_state_update", self.learn_belief_state_update)

        dontcare_value = "dontcare"
        if language == "italian":
            dontcare_value = "non importa"
        if language == "german":
            dontcare_value = "es ist egal"

        for slot_name in slots:
            if dontcare_value not in dialogue_ontology[slot_name] and slot_name != "request":
                # accounting for all slot values and two special values, dontcare and NONE).
                # "Find me food in south part of the city"
                dialogue_ontology[slot_name].append(dontcare_value)
            for value in dialogue_ontology[slot_name]:
                value = str(value)
                if u" " not in value and value not in word_vectors:
                    word_vectors[str(value)] = xavier_vector(str(value))
                    print("-- Generating word vector for:", value.encode("utf-8"), ":::",
                          numpy.sum(word_vectors[value]))  # slot value should have their embedding

        # add up multi-word word values to get their representation: this could be duplicate of previous hard-coding
        for slot in dialogue_ontology.keys():
            if " " in slot:
                slot = str(slot)
                word_vectors[slot] = numpy.zeros((word_vector_size,), dtype="float32")
                constituent_words = slot.split()
                for word in constituent_words:
                    word = str(word)
                    if word in word_vectors:
                        word_vectors[slot] += word_vectors[word]

            for value in dialogue_ontology[slot]:
                if " " in value:
                    value = str(value)
                    word_vectors[value] = numpy.zeros((word_vector_size,), dtype="float32")
                    constituent_words = value.split()
                    for word in constituent_words:
                        word = str(word)
                        if word in word_vectors:
                            word_vectors[value] += word_vectors[word]

        self.use_delex_features = config.get("model", "delex_features")  # what does this mean?

        if self.use_delex_features in ["True", "true"]:
            self.use_delex_features = True
        else:
            self.use_delex_features = False

        self.gpu = config.get("train", "gpu")
        if self.gpu in ["True", "true"]:
            self.device = torch.device("cuda:0")
        self.dtype = torch.float

        # Neural Net Initialisation (keep variables packed so we can move them to either method):
        self.model_variables = {}

        for slot in dialogue_ontology:
            print("Initialisation of model variables for slot:", slot)
            if slot == "request":

                slot_vectors = numpy.zeros((len(dialogue_ontology[slot]), 300), dtype="float32")
                value_vectors = numpy.zeros((len(dialogue_ontology[slot]), 300), dtype="float32")

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                    slot_vectors[value_idx, :] = word_vectors[slot]
                    value_vectors[value_idx, :] = word_vectors[value]

                self.model_variables[slot] = model_definition(word_vector_size, len(dialogue_ontology[slot]),
                                                              slot_vectors, value_vectors, \
                                                              use_delex_features=self.use_delex_features,
                                                              use_softmax=False,
                                                              value_specific_decoder=self.value_specific_decoder,
                                                              learn_belief_state_update=self.learn_belief_state_update,word_vectors_dict=word_vectors)
            else:

                slot_vectors = numpy.zeros((len(dialogue_ontology[slot]) + 1, 300), dtype="float32")  # +1 for None
                value_vectors = numpy.zeros((len(dialogue_ontology[slot]) + 1, 300), dtype="float32")

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                    slot_vectors[value_idx, :] = word_vectors[slot]
                    value_vectors[value_idx, :] = word_vectors[value]

                self.model_variables[slot] = model_definition(word_vector_size, len(dialogue_ontology[slot]),
                                                              slot_vectors, value_vectors,
                                                              use_delex_features=self.use_delex_features, \
                                                              use_softmax=True,
                                                              value_specific_decoder=self.value_specific_decoder,
                                                              learn_belief_state_update=self.learn_belief_state_update,word_vectors_dict=word_vectors)

        self.dialogue_ontology = dialogue_ontology

        self.model_type = model_type
        self.dataset_name = dataset_name

        self.exp_name = exp_name
        self.word_vectors = word_vectors

    def track_utterance(self, current_utterance, req_slot="", conf_slot="", conf_value="", past_belief_state=None):
        """
        Returns a dictionary with predictions for values in the current ontology.
        """
        utterance = current_utterance.decode("utf-8")
        utterance = str(utterance.lower())
        utterance = utterance.replace(u".", u" ")
        utterance = utterance.replace(u",", u" ")
        utterance = utterance.replace(u"?", u" ")
        utterance = utterance.replace(u"-", u" ")
        utterance = utterance.strip()

        if past_belief_state is None:
            past_belief_state = {"food": "none", "area": "none", "price range": "none"}

        utterance = [((utterance, [(utterance, 1.0)]), [req_slot], [conf_slot], [conf_value], past_belief_state)]

        print
        "Testing Utterance: ", utterance

        saver = tf.train.Saver()
        sess = tf.Session()

        prediction_dict = {}
        belief_states = {}
        current_bs = {}

        for slot in self.dialogue_ontology:

            try:
                path_to_load = "models/" + self.model_type + "_en_False_" + \
                               str(self.dataset_name) + "_" + str(slot) + "_" + str(self.exp_name) + "_1.0.ckpt"

                saver.restore(sess, path_to_load)

            except:
                print
                "Can't restore for slot", slot, " - from file:", path_to_load
                return

            belief_state = sliding_window_over_utterance(sess, utterance, self.word_vectors, self.dialogue_ontology,
                                                         self.model_variables, slot)
            belief_states[slot] = belief_state

            # Nikola - red flag, this print could be important.
            predicted_values = return_slot_predictions(belief_state, self.dialogue_ontology[slot], slot, 0.5)
            prediction_dict[slot] = predicted_values

            current_bs[slot] = print_belief_state_woz_informable(self.dialogue_ontology[slot], belief_state,
                                                                 threshold=0.0)  # swap to 0.001 Nikola

        return prediction_dict, current_bs

    def train(self):
        """
        FUTURE: Train the NBT model with new dataset.
        """
        for slot in self.dialogue_ontology.keys():
            print("\n==============  Training the NBT Model for slot", slot, "===============\n")
            stime = time.time()
            train_run(target_language=self.language, override_en_ontology=False, percentage=1.0, model_type="CNN",
                      dataset_name=self.dataset_name, \
                      word_vectors=self.word_vectors, exp_name=self.exp_name, dialogue_ontology=self.dialogue_ontology,
                      model_variables=self.model_variables[slot], target_slot=slot, language=self.language_suffix, \
                      max_epoch=self.max_epoch, batches_per_epoch=self.batches_per_epoch, batch_size=self.batch_size)
            print("\n============== Training this model took", round(time.time() - stime, 1),
                  "seconds. ===================")

    def test_woz(self):

        override_en_ontology = False
        percentage = 1.0

        woz_dialogues, training_turns = load_woz_data(
            "data/" + self.dataset_name + "/" + self.dataset_name + "_test_" + self.language_suffix + ".json",
            self.language, override_en_ontology=False)

        sessions = {}
        saver = tf.train.Saver()

        print
        "WOZ evaluation using language:", self.language, self.language_suffix

        sessions = {}
        saver = tf.train.Saver()

        list_of_belief_states = []

        for model_id in range(0, self.num_models):

            if self.language == "english" or self.language == "en" or override_en_ontology:
                slots_to_load = ["food", "price range", "area", "request"]
            elif self.language == "italian" or self.language == "it":
                slots_to_load = ["cibo", "prezzo", "area", "request"]
            elif self.language == "german" or self.language == "de":
                slots_to_load = ["essen", "preisklasse", "gegend", "request"]

            for load_slot in slots_to_load:
                path_to_load = "./models/" + self.model_type + "_" + self.language_suffix + "_" + str(
                    override_en_ontology) + "_" + \
                               self.dataset_name + "_" + str(load_slot) + "_" + str(self.exp_name) + "_" + str(
                    percentage) + ".ckpt"

                print
                "----------- Loading Model", path_to_load, " ----------------"

                sessions[load_slot] = tf.Session()
                saver.restore(sessions[load_slot], path_to_load)

            evaluated_dialogues, belief_states = track_woz_data(woz_dialogues, self.model_variables, self.word_vectors,
                                                                self.dialogue_ontology, sessions)
            list_of_belief_states.append(belief_states)  # only useful for interpolating.

        results = evaluate_woz(evaluated_dialogues, self.dialogue_ontology)

        json.dump(evaluated_dialogues, open("results/woz_tracking.json", "w"), indent=4)

        print
        json.dumps(results, indent=4)
