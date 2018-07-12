import configparser as ConfigParser
import codecs
import json
import os
import random
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from code.models_py import NBT_model
from code.utils import load_word_vectors, xavier_vector, w2i, i2w, load_woz_data, process_turn_hyp, binary_mask, \
    delexicalise_utterance_values


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

        word_vector_destination = config.get("data", "word_vectors")

        lp = {}
        lp["english"] = u"en"
        lp["german"] = u"de"
        lp["italian"] = u"it"
        self.drop_out = 0.5

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

        self.lr = float(config.get("train", "lr"))
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

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

        word_vector_size = word_vectors["tag-slot"].shape[0]
        self.embedding_dim = word_vector_size

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

        self.unk_token = "UNK"
        if self.unk_token not in word_vectors:
            word_vectors[self.unk_token] = xavier_vector(self.unk_token)

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

        for slot in slots:
            if dontcare_value not in dialogue_ontology[slot] and slot != "request":
                # accounting for all slot values and two special values, dontcare and NONE).
                # "Find me food in south part of the city"
                dialogue_ontology[slot].append(dontcare_value)
                # input("appended dontcare value into slot " + str(slot) + " " + str(len(dialogue_ontology[slot])))
            for value in dialogue_ontology[slot]:
                value = str(value)
                if u" " not in value and value not in word_vectors:
                    word_vectors[str(value)] = xavier_vector(str(value))
                    print("-- Generating word vector for:", value.encode("utf-8"), ":::",
                          np.sum(word_vectors[value]))  # slot value should have their embedding

        # add up multi-word word values to get their representation: this could be duplicate of previous hard-coding
        for slot in dialogue_ontology.keys():
            if " " in slot:
                slot = str(slot)
                word_vectors[slot] = np.zeros((word_vector_size,), dtype="float")
                constituent_words = slot.split()
                for word in constituent_words:
                    word = str(word)
                    if word in word_vectors:
                        word_vectors[slot] += word_vectors[word]

            for value in dialogue_ontology[slot]:
                if " " in value:
                    value = str(value)
                    word_vectors[value] = np.zeros((word_vector_size,), dtype="float")
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
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float
        self.tensor_type = torch.FloatTensor

        # Neural Net Initialisation (keep variables packed so we can move them to either method):
        self.model_variables = {}

        self.w2i_dict = w2i(word_vectors.keys())
        self.i2w_dict = i2w(word_vectors.keys())
        self.drop_out = 0.5

        embedding_value_array = np.array(list(word_vectors.values())).astype(float)
        embedding = torch.nn.Embedding.from_pretrained(self.tensor_type(embedding_value_array, device=self.device))
        # input(embedding)
        embedding.weight.requires_grad = False

        for slot in dialogue_ontology:
            print("Initialisation of model variables for slot: " + slot)
            if slot == "request":

                slot_ids = torch.LongTensor(np.zeros(len(dialogue_ontology[slot]), dtype="int"), device=self.device)
                value_ids = torch.LongTensor(np.zeros(len(dialogue_ontology[slot]), dtype="int"), device=self.device)

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                    slot_ids[value_idx] = self.w2i_dict[slot]
                    value_ids[value_idx] = self.w2i_dict[value]

                self.model_variables[slot] = NBT_model(word_vector_size, len(dialogue_ontology[slot]),
                                                       slot_ids, value_ids,
                                                       use_delex_features=self.use_delex_features,
                                                       use_softmax=False,
                                                       value_specific_decoder=self.value_specific_decoder,
                                                       learn_belief_state_update=self.learn_belief_state_update,
                                                       embedding=embedding, dtype=self.dtype,
                                                       device=self.device, tensor_type=self.tensor_type,
                                                       target_slot=slot,
                                                       value_list=dialogue_ontology[slot], drop_out=self.drop_out)
            else:
                slot_ids = torch.LongTensor(np.zeros(len(dialogue_ontology[slot]) + 1, dtype="int"), device=self.device)
                value_ids = torch.LongTensor(
                    np.zeros(len(dialogue_ontology[slot]) + 1, dtype="int"), device=self.device)  # this includes None

                for value_idx, value in enumerate(dialogue_ontology[slot]):
                    slot_ids[value_idx] = self.w2i_dict[slot]
                    value_ids[value_idx] = self.w2i_dict[value]

                self.model_variables[slot] = NBT_model(word_vector_size, len(dialogue_ontology[slot]),
                                                       slot_ids, value_ids,
                                                       use_delex_features=self.use_delex_features,
                                                       use_softmax=True,
                                                       value_specific_decoder=self.value_specific_decoder,
                                                       learn_belief_state_update=self.learn_belief_state_update,
                                                       embedding=embedding, dtype=self.dtype,
                                                       device=self.device, tensor_type=self.tensor_type,
                                                       target_slot=slot,
                                                       value_list=dialogue_ontology[slot], drop_out=self.drop_out)

        self.dialogue_ontology = dialogue_ontology

        self.model_type = model_type
        self.dataset_name = dataset_name

        self.exp_name = exp_name
        self.embedding = embedding
        self.global_var_asr_count = 1
        self.longest_utterance_length = 40

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

            belief_state = self.sliding_window_over_utterance(sess, utterance, self.word_vectors,
                                                              self.dialogue_ontology,
                                                              self.model_variables, slot)
            belief_states[slot] = belief_state

            # Nikola - red flag, this print could be important.
            predicted_values = self.return_slot_predictions(belief_state, self.dialogue_ontology[slot], slot, 0.5)
            prediction_dict[slot] = predicted_values

            current_bs[slot] = self.print_belief_state_woz_informable(self.dialogue_ontology[slot], belief_state,
                                                                      threshold=0.0)  # swap to 0.001 Nikola

        return prediction_dict, current_bs

    def train_run(self, target_language, override_en_ontology=False, percentage=1.0, model_type="CNN",
                  dataset_name="woz",
                  exp_name=None,
                  dialogue_ontology=None, model_variables=None, target_slot=None, language="en", max_epoch=20,
                  batches_per_epoch=4096,
                  batch_size=256):
        """
        This method trains a slot-specific model on the data and saves the file parameters to a file which can
        then be loaded to do evaluation.
        """

        _, utterances_train2 = load_woz_data(
            "data/" + dataset_name + "/" + dataset_name + "_train_" + language + ".json",
            language)  # utterances_train2 contains a list of tuples,

        utterance_count = len(utterances_train2)  # num_instances

        _, utterances_val2 = load_woz_data(
            "data/" + dataset_name + "/" + dataset_name + "_validate_" + language + ".json", language)
        val_count = len(utterances_val2)

        # utterances_train = utterances_train2 + utterances_val2[0:int(
        #     0.75 * val_count)]  # increment training set. Original split: 600+200+400
        # utterances_val = utterances_val2[
        #                  int(0.75 * val_count):]  # Current split: 750+50+400  # get real index-ed validation utterances

        utterances_train = utterances_train2
        utterances_val = utterances_val2

        print("\nTraining using:", dataset_name, " data - Utterance count:", utterance_count, "target_slot ",
              target_slot)

        # training feature vectors and positive and negative examples list.
        print("Generating data for training set:")
        feature_vectors, positive_examples, negative_examples = self.generate_data(utterances_train, target_slot)

        print("Generating data for validation set:")
        # same for validation (can pre-compute full representation, will be used after each epoch):
        fv_validation, positive_examples_validation, negative_examples_validation = \
            self.generate_data(utterances_val, target_slot)

        val_data = self.generate_examples(target_slot, fv_validation,
                                          positive_examples_validation,
                                          negative_examples_validation)  # get data split
        # val_data is a tuple of (features_full, features_requested_slots, features_confirm_slots,
        # features_confirm_values, features_delex, y_labels, features_previous_state)

        if val_data is None:
            print("val data is none")

        print_mode = False

        # Model training:

        best_f_score = -0.01

        print("\nDoing", batches_per_epoch, "randomly drawn batches of size", batch_size,
              " per epoch. All together for", max_epoch,
              "training epochs ", target_slot)
        start_time = time.time()
        ratio = {}

        for slot in dialogue_ontology:
            if slot not in ratio:
                ratio[slot] = int(batch_size / 2)  # fewer negatives - what does this mean?

        epoch = 0
        last_update = -1

        optimizer = optim.SGD(self.model_variables[target_slot].parameters(), lr=self.lr, momentum=0.9)

        while epoch < max_epoch:

            sys.stdout.flush()

            epoch += 1
            current_epoch_fscore = 0.0
            current_epoch_acc = 0.0

            # if epoch > 1 and target_slot == "request":
            #     return None

            for batch_ind, batch_id in enumerate(range(batches_per_epoch)):
                random_positive_count = ratio[target_slot]  # number of randomly drawn negative example
                random_negative_count = batch_size - random_positive_count  # number of randomly drawn positive example

                batch_data = self.generate_examples(target_slot, feature_vectors,
                                                    positive_examples,
                                                    negative_examples, random_positive_count,
                                                    random_negative_count)

                (batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values,
                 batch_delex, batch_ys, batch_ys_prev) = batch_data

                # input("training model " + str(self.model_variables[target_slot]))
                # forward pass, which loss to define

                batch_ys_pred = self.model_variables[target_slot](
                    batch_data)

                if target_slot == "request":
                    loss = self.MSELoss(batch_ys_pred, batch_ys)
                else:
                    loss = self.CELoss(batch_ys_pred, batch_ys.long())  # TODO check index as label

                print("minibatch ", batch_ind, "loss", loss)
                loss.backward()
                optimizer.step()

                # print(batch_ys.shape, batch_ys_pred.shape, 'aligned? ')

                # [_, cf, cp, cr, ca] = sess.run([train_step, f_score, precision, recall, accuracy],
                #                                feed_dict={x_full: batch_xs_full, \
                #                                           x_delex: batch_delex, \
                #                                           requested_slots: batch_sys_req, \
                #                                           system_act_confirm_slots: batch_sys_conf_slots, \
                #                                           system_act_confirm_values: batch_sys_conf_values, \
                #                                           y_: batch_ys, y_past_state: batch_ys_prev, keep_prob: 0.5})

            # ================================ VALIDATION ==============================================

            epoch_print_step = 1
            if epoch % 5 == 0 or epoch == 1:
                if epoch == 1:
                    print("Epoch", "0", "to", epoch, "took", round(time.time() - start_time, 2), "seconds.")

                else:
                    print("Epoch", epoch - 5, "to", epoch, "took", round(time.time() - start_time, 2), "seconds.")
                    start_time = time.time()

            # param to check (data, dialogue_ontology, \
            #                        positive_examples, negative_examples, print_mode=False, epoch_id=""):

            # val_batch_ys_pred=self.model_variables[target_slot](val_data)

            stime = time.time()
            current_metric = self.model_variables[target_slot].eval_model(val_data)

            # current_f_score = self.model_variables[target_slot].evaluate_model(val_data,
            #                                                                    dialogue_ontology,
            #                                                                    positive_examples_validation,
            #                                                                    negative_examples_validation,
            #                                                                    print_mode=True, epoch_id=epoch + 1)
            # current_metric = current_f_score
            print(" Validation metric for slot at current epoch:", target_slot, " :", round(current_metric, 5),
                  " eval took", round(
                    time.time() - stime, 2), "last update at:", last_update, "/", max_epoch)

            # and if we got a new high score for validation f-score, we need to save the parameters:
            if current_metric > best_f_score:

                last_update = epoch

                # since we are still increasing metric, increase stop epoch count
                if epoch < 100:
                    if int(epoch * 1.5) > max_epoch:
                        max_epoch = int(epoch * 1.5)
                        # print "Increasing max epoch to:", max_epoch
                else:
                    if int(epoch * 1.2) > max_epoch:
                        max_epoch = int(epoch * 1.2)
                        # print "Increasing max epoch to:", max_epoch

                print("\n ====================== New best validation metric:", round(current_metric, 4), \
                      " - saving these parameters. Epoch is:", epoch + 1, "/", max_epoch,
                      "----------------===========  \n")

                best_f_score = current_metric
                path_to_save = "./models/" + model_type + "_" + language + "_" + str(override_en_ontology) + "_" + \
                               str(dataset_name) + "_" + str(target_slot) + "_" + str(exp_name) + "_" + str(
                    percentage) + ".ckpt"

                torch.save(open(path_to_save, "wb"), self.model_variables[target_slot])

        print("The best parameters achieved over all epochs, at validation metric of", round(best_f_score, 4))

    def train(self):
        """
        FUTURE: Train the NBT model with new dataset, slot by slot.
        """
        for slot in self.dialogue_ontology.keys():
            # if slot == "request":
            #     continue
            print("\n==============  Training the NBT Model for slot", slot, "===============\n")
            stime = time.time()
            self.train_run(target_language=self.language, dataset_name=self.dataset_name,
                           exp_name=self.exp_name, dialogue_ontology=self.dialogue_ontology,
                           model_variables=self.model_variables[slot], target_slot=slot, language=self.language_suffix, \
                           max_epoch=self.max_epoch, batches_per_epoch=self.batches_per_epoch,
                           batch_size=self.batch_size)
            print("\n============== Training this slot-specific model took", round(time.time() - stime, 1),
                  "seconds. ===================")

    def generate_data(self, utterances, target_slot):
        """
        Generates a data representation we can subsequently use.

        Let's say negative requests are now - those utterances which express no requestables.
        """

        # utterences: a list of tuples nearly all information in a turn
        feature_vectors = self.extract_feature_vectors(utterances, self.embedding)

        # indexed by slot, these two dictionaries contain lists of positive and negative examples
        # for training each slot. Each list element is (utterance_id, slot_id, value_id)
        positive_examples = {}
        negative_examples = {}

        list_of_slots = [target_slot]  # never used?
        # list_of_slots = dialogue_ontology.keys()  # ["food", "area", "price range", "request"] ???

        for slot_idx, slot in enumerate(list_of_slots):

            positive_examples[slot] = []
            negative_examples[slot] = []

            for utterance_idx, utterance in enumerate(utterances):

                slot_expressed_in_utterance = False

                # utterance[4] is the current label
                # utterance[5] is the previous one

                for (slotA, valueA) in utterance[4]:
                    if slotA == slot and (valueA != "none" and valueA != []):
                        slot_expressed_in_utterance = True

                        # if slot == "request":
                        #    print slotA, valueA, utterance, utterance[4]

                if slot != "request":

                    for value_idx, value in enumerate(self.dialogue_ontology[slot]):

                        if (slot, value) in utterance[
                            4]:  # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                            positive_examples[slot].append((utterance_idx, utterance, value_idx))
                            # print "POS:", utterance_idx, utterance, value_idx, value
                        else:
                            if not slot_expressed_in_utterance:
                                negative_examples[slot].append((utterance_idx, utterance, value_idx))
                                # print "NEG:", utterance_idx, utterance, value_idx, value

                elif slot == "request":

                    if not slot_expressed_in_utterance:
                        negative_examples[slot].append((utterance_idx, utterance, []))
                        # print utterance[0][0], utterance[4]
                    else:
                        values_expressed = []
                        for value_idx, value in enumerate(self.dialogue_ontology[slot]):
                            if (slot, value) in utterance[
                                4]:  # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                                values_expressed.append(value_idx)

                        positive_examples[slot].append((utterance_idx, utterance, values_expressed))

        return feature_vectors, positive_examples, negative_examples

    def extract_feature_vectors(self, utterances, ngram_size=3, use_asr=False,
                                use_transcription_in_training=False):
        """
        This method returns feature vectors for all dialogue utterances.
        It returns a tuple of lists, where each list consists of all feature vectors for ngrams of that length. e.g. up to trigram

        This method doesn't care about the labels: other methods assign actual or fake labels later on.

        This can run on any size, including a single utterance.

        """
        utterance_count = len(utterances)

        ngram_feature_vectors = []
        requested_slot_vectors = []
        confirm_slots = []
        confirm_values = []

        # let index 6 denote full FV (for conv net). Why is the shape like this?
        for j in range(0, utterance_count):
            ngram_feature_vectors.append(
                np.zeros((self.longest_utterance_length * self.embedding_dim,), dtype="float32"))

        print("total utterances len " + str(len(utterances)))

        for idx, utterance in enumerate(utterances):

            # full_fv = torch.Tensor(np.zeros((self.longest_utterance_length * self.embedding_dim,), dtype="float32"))
            # np.zeros((self.longest_utterance_length * self.embedding_dim,), dtype="float32")

            # use_asr = True

            if use_asr:
                full_asr = utterances[idx][0][1]  # just use ASR, [(hypo-1, 1.0), (hypo-2, 0.8)]
            else:
                full_asr = [(utterances[idx][0][0], 1.0)]  # else create (transcription, 1.0)

            # encode last system utterance

            requested_slots = utterances[idx][1]

            current_requested_vector = torch.Tensor(np.zeros((self.embedding_dim,), dtype="float32"),
                                                    device=self.device)

            # requested_slot="area"
            # print("w2i contains this word " + str(str(requested_slot) in self.w2i_dict.keys()))
            # print("get its embedding as "+str(self.embedding(torch.LongTensor([self.w2i_dict[str(requested_slot)]]))))
            #
            # input()

            # print(self.embedding(
            #             torch.LongTensor([self.w2i_dict[str("place")]])).squeeze(0).shape)
            #
            # print(current_requested_vector.shape)
            # input("shape valid match?")

            for requested_slot in requested_slots:
                if requested_slot != "":
                    # input("In this example, full_asr is " + str(full_asr) + " requested_slot" + str(requested_slots))
                    current_requested_vector += self.embedding(
                        torch.LongTensor([self.w2i_dict[str(requested_slot)]], device=self.device)).squeeze(
                        0)  # add all requests up

            requested_slot_vectors.append(current_requested_vector)

            curr_confirm_slots = utterances[idx][2]
            curr_confirm_values = utterances[idx][3]

            # for ind,cfm_slot in enumerate(curr_confirm_slots):
            #     if cfm_slot!="":
            #         input("In this example, full_asr is " + str(full_asr) + " cfm_slot: " + str(cfm_slot)+" value: "+curr_confirm_values[ind])
            #         break

            current_conf_slot_vector = torch.Tensor(np.zeros((self.embedding_dim,), dtype="float32"),
                                                    device=self.device)
            current_conf_value_vector = torch.Tensor(np.zeros((self.embedding_dim,), dtype="float32"),
                                                     device=self.device)

            confirmation_count = len(curr_confirm_slots)

            for sub_idx in range(0, confirmation_count):
                current_cslot = curr_confirm_slots[sub_idx]
                current_cvalue = curr_confirm_values[sub_idx]

                if current_cslot != "" and current_cvalue != "":  # iff valid (slot,value) pair
                    if " " not in current_cslot:
                        current_conf_slot_vector += self.embedding(
                            torch.LongTensor([self.w2i_dict[str(current_cslot)]])).squeeze(0)
                    else:
                        words_in_example = current_cslot.split()
                        for cword in words_in_example:
                            current_conf_slot_vector += self.embedding(
                                torch.LongTensor([self.w2i_dict[str(cword)]])).squeeze(0)

                    if " " not in current_cvalue:
                        current_conf_value_vector += self.embedding(
                            torch.LongTensor([self.w2i_dict[str(current_cvalue)]])).squeeze(0)
                    else:
                        words_in_example = current_cvalue.split()
                        for cword in words_in_example:
                            current_conf_value_vector += self.embedding(
                                torch.LongTensor([self.w2i_dict[str(cword)]])).squeeze(0)

            confirm_slots.append(current_conf_slot_vector)
            confirm_values.append(current_conf_value_vector)

            asr_weighted_feature_vectors = []

            asr_count = self.global_var_asr_count  # how many hypothesis do we consider
            asr_mass = 0.0  # total mass

            for idx1 in range(0, asr_count):
                asr_mass += full_asr[idx1][
                    1]  # add up mass over all hypothesis, + full_asr[1][1] + full_asr[2][1] + full_asr[3][1] + full_asr[4][1]

            if use_transcription_in_training:
                transcription_mass = asr_mass - full_asr[asr_count - 1][1]
                extra_example = (
                    utterances[idx][0][0], transcription_mass)  # extra_example enjoys the sum of all other weights
                full_asr[asr_count - 1] = extra_example
                asr_mass = 2 * transcription_mass

            for (c_example, asr_coeff) in full_asr[0:asr_count]:

                # print c_example, asr_coeff

                full_fv = torch.Tensor(
                    np.zeros((self.longest_utterance_length * self.embedding_dim,), dtype="float32"),
                    device=self.device)
                if c_example != "":
                    # print c_example
                    words_utterance = process_turn_hyp(c_example, "en")  # cleaned text: lowercase and normalize
                    words_utterance = words_utterance.split()

                    for word_idx, word in enumerate(words_utterance):

                        word = str(word)

                        if word not in self.w2i_dict:
                            this_vector = self.embedding(
                                torch.LongTensor([self.w2i_dict[self.unk_token]], device=self.device))
                            print("Looping over Utterance and generating data: Generating UNK word vector for",
                                  word.encode('utf-8'))

                        try:
                            full_fv[
                            word_idx * self.embedding_dim: (word_idx + 1) * self.embedding_dim] = self.embedding(
                                torch.LongTensor([self.w2i_dict[str(word)]], device=self.device))
                        except:
                            print("Something off with word:", word, word in self.w2i_dict)

                asr_weighted_feature_vectors.append(
                    full_fv.view(self.longest_utterance_length, self.embedding_dim))  # reshape tensor

            if len(asr_weighted_feature_vectors) != asr_count:
                print("Please verify: length of weighted vectors is:", len(asr_weighted_feature_vectors))

                # list of [40, 300] into [len_list * 40, 300]
            # print len(asr_weighted_feature_vectors), asr_weighted_feature_vectors[0].shape
            # ngram_feature_vectors[idx] = np.concatenate(asr_weighted_feature_vectors, axis=0)

            # TODO: this is hard-coding, that assumes use_asr=False
            ngram_feature_vectors[idx] = asr_weighted_feature_vectors[0]

        list_of_features = []
        use_external_representations = False

        for idx in range(0, utterance_count):
            list_of_features.append((ngram_feature_vectors[idx],
                                     requested_slot_vectors[idx],
                                     confirm_slots[idx],
                                     confirm_values[idx],
                                     ))

        return list_of_features  # a list of tuples (u, t_q, t_s, t_v), that could be used by equation (1) (9) (10)

    def generate_examples(self, target_slot, feature_vectors,
                          positive_examples, negative_examples, positive_count=None, negative_count=None):
        """
        This method returns a minibatch of positive_count examples followed by negative_count examples.
        If these two are not set, it creates the full dataset (used for validation and test).
        It returns: (features_unigram, features_bigram, features_trigram, features_slot,
                     features_values, y_labels) - all we need to pass to train.
        """

        # total number of positive and negative examples.
        pos_example_count = len(positive_examples[target_slot])
        neg_example_count = len(negative_examples[target_slot])

        if target_slot != "request":
            label_count = len(self.dialogue_ontology[target_slot]) + 1  # NONE
            print("target_slot label_count is " + str(label_count))
        else:
            label_count = len(self.dialogue_ontology[target_slot])

        # doing_validation = False   # for validation?
        if positive_count is None:
            positive_count = pos_example_count
            # doing_validation = True
        if negative_count is None:
            negative_count = neg_example_count
            # doing_validation = True

        if pos_example_count == 0 or positive_count == 0 or negative_count == 0 or neg_example_count == 0:
            print("#### SKIPPING TRAINING (NO DATA): ", target_slot, pos_example_count, positive_count,
                  neg_example_count, negative_count)
            return None

        positive_indices = []
        negative_indices = []

        if positive_count > 0:  # select only positive_count number of instances
            positive_indices = np.random.choice(pos_example_count, positive_count)
        else:
            print(target_slot, positive_count, negative_count)

        if negative_count > 0:
            negative_indices = np.random.choice(neg_example_count, negative_count)  # with replacement

        examples = []
        labels = []
        prev_labels = []

        for idx in positive_indices:
            examples.append(positive_examples[target_slot][idx])
        if negative_count > 0:
            for idx in negative_indices:
                examples.append(negative_examples[target_slot][idx])

        value_count = len(self.dialogue_ontology[target_slot])

        # each element of this array is (xs_unigram, xs_bigram, xs_trigram, fv_slot, fv_value):
        features_requested_slots = []
        features_confirm_slots = []
        features_confirm_values = []
        features_slot = []
        features_values = []
        features_full = []
        features_delex = []
        features_previous_state = []

        # feature vector of the used slot:
        candidate_slot = self.embedding(torch.LongTensor([self.w2i_dict[str(target_slot)]], device=self.device))

        # now go through all examples (positive followed by negative).
        for idx_example, example in enumerate(examples):

            (utterance_idx, utterance, value_idx) = example
            utterance_fv = feature_vectors[utterance_idx]  # self.longest_ * self.embedding_dim

            # prev belief state is in utterance[5]
            prev_belief_state = utterance[5]

            if idx_example < positive_count:
                if target_slot != "request":
                    labels.append(value_idx)  # includes dontcare
                else:
                    labels.append(
                        binary_mask(value_idx, len(self.dialogue_ontology["request"])))  # appends a one-hot tensor
            else:  # negative example
                if target_slot != "request":
                    labels.append(
                        value_count)  # NONE - for this we need to make sure to not include utterances which express this slot
                else:
                    labels.append([])  # wont ever use this

            # handling of previous labels:
            if target_slot != "request":
                prev_labels.append(prev_belief_state[target_slot])

            # for now, we just deal with the utterance, and not with WOZ data.
            # TODO: need to get a series of delexicalised vectors, one for each value.

            delex_features = delexicalise_utterance_values(utterance[0][0], target_slot,
                                                           self.dialogue_ontology[target_slot])
            # read pure text, generate delex_features

            features_full.append(utterance_fv[0])  # text word vectors
            features_requested_slots.append(utterance_fv[1])
            features_confirm_slots.append(utterance_fv[2])
            features_confirm_values.append(utterance_fv[3])
            features_delex.append(delex_features)

            prev_belief_state_vector = torch.Tensor(np.zeros((label_count,), dtype="float32"), device=self.device)

            if target_slot != "request":

                prev_value = prev_belief_state[target_slot]

                if prev_value == "none" or prev_value not in self.dialogue_ontology[target_slot]:
                    prev_belief_state_vector[label_count - 1] = 1  # no value
                else:
                    prev_belief_state_vector[self.dialogue_ontology[target_slot].index(prev_value)] = 1

            features_previous_state.append(prev_belief_state_vector)

        # TODO: change to torch Tensor
        features_requested_slots = torch.stack(features_requested_slots)
        features_confirm_slots = torch.stack(features_confirm_slots)
        features_confirm_values = torch.stack(features_confirm_values)
        features_full = torch.stack(features_full)
        features_delex = torch.stack(features_delex)
        features_previous_state = torch.stack(features_previous_state)

        for idx in range(0, positive_count):
            if target_slot != "request":
                y_labels = torch.Tensor(np.zeros(positive_count + negative_count, dtype="float32"), device=self.device)
                y_labels[idx] = labels[idx]
            else:
                y_labels = torch.Tensor(np.zeros((positive_count + negative_count, label_count), dtype="float32"),
                                        device=self.device)
                y_labels[idx, :] = labels[idx]

        if target_slot != "request":
            y_labels[
            positive_count:] = label_count - 1  # NONE, 0-indexing, starting from index positive_count, all are negative cases
            # input(label_count-1)

        # if target_slot == "request" then all zero?

        return (features_full, features_requested_slots, features_confirm_slots, \
                features_confirm_values, features_delex, y_labels, features_previous_state)

    def test_woz(self):
        override_en_ontology = False
        percentage = 1.0

        woz_dialogues, _ = load_woz_data(
            "data/" + self.dataset_name + "/" + self.dataset_name + "_test_" + self.language_suffix + ".json",
            self.language, override_en_ontology=False)

        print("WOZ evaluation using language:", self.language, self.language_suffix)

        list_of_belief_states = []

        models = dict()

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

                models[load_slot] = torch.load(path_to_load)

            evaluated_dialogues, belief_states = self.track_woz_data(woz_dialogues, self.model_variables,
                                                                     self.word_vectors,
                                                                     self.dialogue_ontology, models)
            list_of_belief_states.append(belief_states)  # only useful for interpolating.

        results = self.evaluate_woz(evaluated_dialogues, self.dialogue_ontology)

        json.dump(evaluated_dialogues, open("results/woz_tracking.json", "w"), indent=4)

        print(json.dumps(results, indent=4))
        return

    def track_woz_data(self, dialogues, model_variables, word_vectors, dialogue_ontology, models):
        """
        This method evaluates the WOZ dialogues.
        """
        evaluated_dialogues = []
        list_of_belief_states = []

        dialogue_count = len(dialogues)
        input("DIALOGUE COUNT: ", dialogue_count)

        for idx in range(0, dialogue_count):

            if idx % 100 == 0 and (dialogue_count == 400):  # progress for test
                print(idx, "/", dialogue_count, "done.")

            evaluated_dialogue, belief_states = self.track_dialogue_woz(model_variables, word_vectors,
                                                                        dialogue_ontology,
                                                                        dialogues[idx], models)

            evaluated_dialogues.append(evaluated_dialogue)
            list_of_belief_states.append(belief_states)

        dialogue_count = len(evaluated_dialogues)
        indexed_dialogues = []
        for d_idx in range(0, dialogue_count):
            new_dialogue = {}
            new_dialogue["dialogue_idx"] = d_idx
            new_dialogue["dialogue"] = evaluated_dialogues[d_idx]
            indexed_dialogues.append(new_dialogue)

        return indexed_dialogues, list_of_belief_states

    def track_dialogue_woz(self, model_variables, word_vectors, dialogue_ontology, woz_dialogue, models):
        """
        This produces a list of belief states predicted for the given WOZ dialogue.
        """

        prev_belief_states = {}
        belief_states = {}  # for each slot, a list of np arrays.

        turn_count = len(woz_dialogue)
        # print "Turn count:", turn_count

        slots_to_track = list(set(dialogue_ontology.keys()) & set(models.keys()))

        for slot in slots_to_track:
            belief_states[slot] = {}
            if slot != "request":
                value_count = len(dialogue_ontology[slot]) + 1
                prev_belief_states[slot] = np.zeros((value_count,), dtype="float32")

        predictions_for_dialogue = []

        # to be able to combine predictions, we must also return the belief states for each turn. So for each turn, a dictionary indexed by slot values which points to the distribution.
        list_of_belief_states = []

        # print woz_dialogue

        for idx, trans_and_req_and_label_and_currlabel in enumerate(woz_dialogue):

            list_of_belief_states.append({})

            current_bs = {}

            for slot in slots_to_track:

                if type(model_variables) is dict:
                    mx = model_variables[slot]
                else:
                    mx = model_variables

                # print trans_and_req_and_label_and_currlabel

                (transcription_and_asr, req_slot, conf_slot, conf_value, label,
                 prev_belief_state) = trans_and_req_and_label_and_currlabel

                if idx == 0 or slot == "request":
                    # this should put empty belief state
                    example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_belief_state)]
                else:
                    # and this has the previous prediction, the one we just made in the previous iteration. We do not want to use the right one, the one used for training.
                    example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_bs)]

                    # print example

                transcription = transcription_and_asr[0]
                asr = transcription_and_asr[1]

                if slot == "request":

                    updated_belief_state = self.sliding_window_over_utterance(models[slot], example, word_vectors,
                                                                              dialogue_ontology, mx, slot,
                                                                              print_mode=False)
                    # updated_belief_state[updated_belief_state < 0.5] = 0
                    list_of_belief_states[idx]["request"] = updated_belief_state

                else:
                    updated_belief_state = self.sliding_window_over_utterance(models[slot], example, word_vectors,
                                                                              dialogue_ontology, mx, slot,
                                                                              print_mode=False)
                    # updated_belief_state = softmax(updated_belief_state)
                    # updated_belief_state = update_belief_state(prev_belief_states[slot], new_belief_state)
                    prev_belief_states[slot] = updated_belief_state
                    list_of_belief_states[idx][slot] = updated_belief_state

                for idx_value, value in enumerate(dialogue_ontology[slot]):
                    if slot in "request":
                        current_bs[slot] = self.print_belief_state_woz_requestables(dialogue_ontology[slot],
                                                                                    updated_belief_state, threshold=0.5)
                    else:
                        current_bs[slot] = self.print_belief_state_woz_informable(dialogue_ontology[slot],
                                                                                  updated_belief_state,
                                                                                  threshold=0.01)  # swap to 0.001 Nikola
                        #   print idx, slot, current_bs[slot], current_bs

            prev_bs = deepcopy(current_bs)

            trans_plus_sys = "User: " + transcription
            # + req_slot, conf_slot, conf_value
            if req_slot[0] != "":
                trans_plus_sys += "    System Request: " + str(req_slot)

            if conf_slot[0] != "":
                trans_plus_sys += "    System Confirm: " + str(conf_slot) + " " + str(conf_value)

            trans_plus_sys += "   ASR: " + str(asr)

            predictions_for_dialogue.append((trans_plus_sys, {"True State": label}, {"Prediction": current_bs}))

        return predictions_for_dialogue, list_of_belief_states

    def evaluate_woz(self, evaluated_dialogues, dialogue_ontology):
        """
        Given a list of (transcription, correct labels, predicted labels), this measures joint goal (as in Matt's paper),
        and f-scores, as presented in Shawn's NIPS paper.

        Assumes request is always there in the ontology.
        """
        print_mode = True
        informable_slots = list(
            set(["food", "area", "price range", "prezzo", "cibo", "essen", "preisklasse", "gegend"]) & set(
                dialogue_ontology.keys()))
        dialogue_count = len(evaluated_dialogues)
        if "request" in dialogue_ontology:
            req_slots = [str("req_" + x) for x in dialogue_ontology["request"]]
            requestables = ["request"]
        else:
            req_slots = []
            requestables = []
        # print req_slots

        true_positives = {}
        false_negatives = {}
        false_positives = {}

        req_match = 0.0
        req_full_turn_count = 0.0

        req_acc_total = 0.0  # number of turns which express requestables
        req_acc_correct = 0.0

        for slot in dialogue_ontology:
            true_positives[slot] = 0
            false_positives[slot] = 0
            false_negatives[slot] = 0

        for value in requestables + req_slots + ["request"]:
            true_positives[value] = 0
            false_positives[value] = 0
            false_negatives[value] = 0

        correct_turns = 0  # when there is at least one informable, do all of them match?
        incorrect_turns = 0  # when there is at least one informable, if any are different.

        slot_correct_turns = {}
        slot_incorrect_turns = {}

        for slot in informable_slots:
            slot_correct_turns[slot] = 0.0
            slot_incorrect_turns[slot] = 0.0

        dialogue_joint_metrics = []
        dialogue_req_metrics = []

        dialogue_slot_metrics = {}

        for slot in informable_slots:
            dialogue_slot_metrics[slot] = []

        for idx in range(0, dialogue_count):

            dialogue = evaluated_dialogues[idx]["dialogue"]
            # print dialogue

            curr_dialogue_goal_joint_total = 0.0  # how many turns have informables
            curr_dialogue_goal_joint_correct = 0.0

            curr_dialogue_goal_slot_total = {}  # how many turns in current dialogue have specific informables
            curr_dialogue_goal_slot_correct = {}  # and how many of these are correct

            for slot in informable_slots:
                curr_dialogue_goal_slot_total[slot] = 0.0
                curr_dialogue_goal_slot_correct[slot] = 0.0

            creq_tp = 0.0
            creq_fn = 0.0
            creq_fp = 0.0
            # to compute per-dialogue f-score for requestables

            for turn in dialogue:

                # first update full requestable

                req_full_turn_count += 1.0

                if requestables:

                    if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                        req_match += 1.0

                    if len(turn[1]["True State"]["request"]) > 0:
                        req_acc_total += 1.0

                        if compare_request_lists(turn[1]["True State"]["request"], turn[2]["Prediction"]["request"]):
                            req_acc_correct += 1.0

                # per dialogue requestable metrics
                if requestables:

                    true_requestables = turn[1]["True State"]["request"]
                    predicted_requestables = turn[2]["Prediction"]["request"]

                    for each_true_req in true_requestables:
                        if each_true_req in dialogue_ontology["request"] and each_true_req in predicted_requestables:
                            true_positives["request"] += 1
                            creq_tp += 1.0
                            true_positives["req_" + each_true_req] += 1
                        elif each_true_req in dialogue_ontology["request"]:
                            false_negatives["request"] += 1
                            false_negatives["req_" + each_true_req] += 1
                            creq_fn += 1.0
                            # print "FN:", turn[0], "---", true_requestables, "----", predicted_requestables

                    for each_predicted_req in predicted_requestables:
                        # ignore matches, already counted, now need just negatives:
                        if each_predicted_req not in true_requestables:
                            false_positives["request"] += 1
                            false_positives["req_" + each_predicted_req] += 1
                            creq_fp += 1.0
                            # print "-- FP:", turn[0], "---", true_requestables, "----", predicted_requestables

                # print turn
                inf_present = {}
                inf_correct = {}

                for slot in informable_slots:
                    inf_present[slot] = False
                    inf_correct[slot] = True

                informable_present = False
                informable_correct = True

                for slot in informable_slots:

                    try:
                        true_value = turn[1]["True State"][slot]
                        predicted_value = turn[2]["Prediction"][slot]
                    except:

                        print
                        "PROBLEM WITH", turn, "slot:", slot, "inf slots", informable_slots

                    if true_value != "none":
                        informable_present = True
                        inf_present[slot] = True

                    if true_value == predicted_value:  # either match or none, so not incorrect
                        if true_value != "none":
                            true_positives[slot] += 1
                    else:
                        if true_value == "none":
                            false_positives[slot] += 1
                        elif predicted_value == "none":
                            false_negatives[slot] += 1
                        else:
                            # spoke to Shawn - he does this as false negatives for now - need to think about how we evaluate it properly.
                            false_negatives[slot] += 1

                        informable_correct = False
                        inf_correct[slot] = False

                if informable_present:

                    curr_dialogue_goal_joint_total += 1.0

                    if informable_correct:
                        correct_turns += 1
                        curr_dialogue_goal_joint_correct += 1.0
                    else:
                        incorrect_turns += 1

                for slot in informable_slots:
                    if inf_present[slot]:
                        curr_dialogue_goal_slot_total[slot] += 1.0

                        if inf_correct[slot]:
                            slot_correct_turns[slot] += 1.0
                            curr_dialogue_goal_slot_correct[slot] += 1.0
                        else:
                            slot_incorrect_turns[slot] += 1.0

            # current dialogue requestables

            if creq_tp + creq_fp > 0.0:
                creq_precision = creq_tp / (creq_tp + creq_fp)
            else:
                creq_precision = 0.0

            if creq_tp + creq_fn > 0.0:
                creq_recall = creq_tp / (creq_tp + creq_fn)
            else:
                creq_recall = 0.0

            if creq_precision + creq_recall == 0:
                if creq_tp == 0 and creq_fn == 0 and creq_fn == 0:
                    # no requestables expressed, special value
                    creq_fscore = -1.0
                else:
                    creq_fscore = 0.0  # none correct but some exist
            else:
                creq_fscore = (2 * creq_precision * creq_recall) / (creq_precision + creq_recall)

            dialogue_req_metrics.append(creq_fscore)

            # and current dialogue informables:

            for slot in informable_slots:
                if curr_dialogue_goal_slot_total[slot] > 0:
                    dialogue_slot_metrics[slot].append(
                        float(curr_dialogue_goal_slot_correct[slot]) / curr_dialogue_goal_slot_total[slot])
                else:
                    dialogue_slot_metrics[slot].append(-1.0)

            if informable_slots:
                if curr_dialogue_goal_joint_total > 0:
                    current_dialogue_joint_metric = float(
                        curr_dialogue_goal_joint_correct) / curr_dialogue_goal_joint_total
                    dialogue_joint_metrics.append(current_dialogue_joint_metric)
                else:
                    # should not ever happen when all slots are used, but for validation we might not have i.e. area mentioned
                    dialogue_joint_metrics.append(-1.0)

        if informable_slots:
            goal_joint_total = float(correct_turns) / float(correct_turns + incorrect_turns)

        slot_gj = {}

        total_true_positives = 0
        total_false_negatives = 0
        total_false_positives = 0

        precision = {}
        recall = {}
        fscore = {}

        # FSCORE for each requestable slot:
        if requestables:
            add_req = ["request"] + req_slots
        else:
            add_req = []

        for slot in informable_slots + add_req:

            if slot not in ["request"] and slot not in req_slots:
                total_true_positives += true_positives[slot]
                total_false_positives += false_positives[slot]
                total_false_negatives += false_negatives[slot]

            precision_denominator = (true_positives[slot] + false_positives[slot])

            if precision_denominator != 0:
                precision[slot] = float(true_positives[slot]) / precision_denominator
            else:
                precision[slot] = 0

            recall_denominator = (true_positives[slot] + false_negatives[slot])

            if recall_denominator != 0:
                recall[slot] = float(true_positives[slot]) / recall_denominator
            else:
                recall[slot] = 0

            if precision[slot] + recall[slot] != 0:
                fscore[slot] = (2 * precision[slot] * recall[slot]) / (precision[slot] + recall[slot])
                print
                "REQ - slot", slot, round(precision[slot], 3), round(recall[slot], 3), round(fscore[slot], 3)
            else:
                fscore[slot] = 0

            total_count_curr = true_positives[slot] + false_negatives[slot] + false_positives[slot]

            # if "req" in slot:
            # if slot in ["area", "food", "price range", "request"]:
            # print "Slot:", slot, "Count:", total_count_curr, true_positives[slot], false_positives[slot], false_negatives[slot], "[Precision, Recall, Fscore]=", round(precision[slot], 2), round(recall[slot], 2), round(fscore[slot], 2)
            # print "Slot:", slot, "TP:", true_positives[slot], "FN:", false_negatives[slot], "FP:", false_positives[slot]

        if requestables:

            requested_accuracy_all = req_match / req_full_turn_count

            if req_acc_total != 0:
                requested_accuracy_exist = req_acc_correct / req_acc_total
            else:
                requested_accuracy_exist = 1.0

            slot_gj["request"] = round(requested_accuracy_exist, 3)
            # slot_gj["requestf"] = round(fscore["request"], 3)

        for slot in informable_slots:
            slot_gj[slot] = round(
                float(slot_correct_turns[slot]) / float(slot_correct_turns[slot] + slot_incorrect_turns[slot]), 3)

        # NIKOLA TODO: will be useful for goal joint
        if len(informable_slots) == 3:
            # print "\n\nGoal Joint: " + str(round(goal_joint_total, 3)) + "\n"
            slot_gj["joint"] = round(goal_joint_total, 3)

        return slot_gj

    def print_belief_state_woz_informable(self, curr_values, distribution, threshold):
        """
        Returns the top one if it is above threshold.
        """
        max_value = "none"
        max_score = 0.0
        total_value = 0.0

        for idx, value in enumerate(curr_values):

            total_value += distribution[idx]

            if distribution[idx] >= threshold:

                if distribution[idx] >= max_score:
                    max_value = value
                    max_score = distribution[idx]

        if max_score >= (1.0 - total_value):
            return max_value
        else:
            return "none"

    def print_belief_state_woz_requestables(self, curr_values, distribution, threshold):
        """
        Returns the top one if it is above threshold.
        """
        requested_slots = []

        # now we just print to JSON file:
        for idx, value in enumerate(curr_values):

            if distribution[idx] >= threshold:
                requested_slots.append(value)

        return requested_slots

    def print_slot_predictions(self, distribution, slot_values, target_slot, threshold=0.05):
        """
        Prints all the activated slot values for the provided predictions
        """

        predicted_values = []
        for idx, value in enumerate(slot_values):
            if distribution[idx] >= threshold:
                predicted_values += ((value, round(distribution[idx], 2)))  # , round(post_sf[idx], 2) ))

        print
        "Predictions for", str(target_slot + ":"), predicted_values

        return predicted_values

    def return_slot_predictions(self, distribution, slot_values, target_slot, threshold=0.05):
        """
        Prints all the activated slot values for the provided predictions
        """
        predicted_values = []
        for idx, value in enumerate(slot_values):
            if distribution[idx] >= threshold:
                predicted_values += ((value, round(distribution[idx], 2)))  # , round(post_sf[idx], 2) ))

        return predicted_values

    def sliding_window_over_utterance(self, model, utterance, word_vectors, dialogue_ontology, model_variables,
                                      target_slot,
                                      print_mode=True):
        """
        """

        if type(model_variables) is dict:
            model_variables = model_variables[target_slot]

        list_of_outputs = self.test_utterance(model, utterance, word_vectors, dialogue_ontology, model_variables,
                                              target_slot,
                                              print_mode)
        belief_state = list_of_outputs[0]

        return belief_state

    def test_utterance(self, model, utterances, word_vectors, dialogue_ontology, model_variables, target_slot,
                       do_print=True):
        """
        Returns a list of belief states, to be weighted later.
        """

        potential_values = dialogue_ontology[target_slot]

        if target_slot == "request":
            value_count = len(potential_values)
        else:
            value_count = len(potential_values) + 1

        # should be a list of features for each ngram supplied.
        fv_tuples = self.extract_feature_vectors(utterances, word_vectors, use_asr=True)
        utterance_count = len(utterances)

        belief_state = np.zeros((value_count,), dtype="float32")

        # accumulators
        slot_values = []
        candidate_values = []
        delexicalised_features = []
        fv_full = []
        fv_sys_req = []
        fv_conf_slot = []
        fv_conf_val = []
        features_previous_state = []

        for idx_hyp, extracted_fv in enumerate(fv_tuples):

            current_utterance = utterances[idx_hyp][0][0]

            prev_belief_state = utterances[idx_hyp][4]

            prev_belief_state_vector = np.zeros((value_count,), dtype="float32")

            if target_slot != "request":

                prev_value = prev_belief_state[target_slot]

                if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                    prev_belief_state_vector[value_count - 1] = 1
                else:
                    prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

            features_previous_state.append(prev_belief_state_vector)

            (full_utt, sys_req, conf_slot, conf_value) = extracted_fv

            delex_vector = delexicalise_utterance_values(current_utterance, target_slot, dialogue_ontology[target_slot])

            fv_full.append(full_utt)
            delexicalised_features.append(delex_vector)
            fv_sys_req.append(sys_req)
            fv_conf_slot.append(conf_slot)
            fv_conf_val.append(conf_value)

        slot_values = np.array(slot_values)
        candidate_values = np.array(candidate_values)
        delexicalised_features = np.array(
            delexicalised_features)  # will be [batch_size, label_size, longest_utterance_length, vector_dimension]

        fv_sys_req = np.array(fv_sys_req)
        fv_conf_slot = np.array(fv_conf_slot)
        fv_conf_val = np.array(fv_conf_val)
        fv_full = np.array(fv_full)
        features_previous_state = np.array(features_previous_state)

        keep_prob, x_full, x_delex, \
        requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
        f_score, precision, recall, num_true_positives, \
        num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
        true_positives, train_step, update_coefficient = model_variables

        distribution, update_coefficient_load = sess.run([y, update_coefficient],
                                                         feed_dict={x_full: fv_full, x_delex: delexicalised_features, \
                                                                    requested_slots: fv_sys_req, \
                                                                    system_act_confirm_slots: fv_conf_slot,
                                                                    y_past_state: features_previous_state,
                                                                    system_act_confirm_values: fv_conf_val, \
                                                                    keep_prob: 1.0})

        belief_state = distribution[:, 0]

        current_start_idx = 0
        list_of_belief_states = []

        for idx in range(0, utterance_count):
            current_distribution = distribution[idx, :]
            list_of_belief_states.append(current_distribution)

        if do_print:
            print_slot_predictions(list_of_belief_states[0], potential_values, target_slot, threshold=0.1)

        if len(list_of_belief_states) == 1:
            return [list_of_belief_states[0]]

        return list_of_belief_states
