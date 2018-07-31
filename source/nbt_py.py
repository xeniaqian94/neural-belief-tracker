import json
import math
import sys
from copy import deepcopy
from source.neural_belief_tracker import NeuralBeliefTracker
import torch

global_var_asr_count = 1

lp = {}
lp["english"] = u"en"
lp["german"] = u"de"
lp["italian"] = u"it"
lp["russian"] = u"ru"
lp["sh"] = u"sh"
lp["bulgarian"] = u"bg"
lp["polish"] = u"pl"
lp["spanish"] = u"es"
lp["french"] = u"fr"
lp["portuguese"] = u"pt"
lp["swedish"] = u"sv"
lp["dutch"] = u"nl"


def compare_request_lists(list_a, list_b):
    if len(list_a) != len(list_b):
        return False

    list_a.sort()
    list_b.sort()

    for idx in range(0, len(list_a)):
        if list_a[idx] != list_b[idx]:
            return False

    return True


def evaluate_woz(evaluated_dialogues, dialogue_ontology):
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
                current_dialogue_joint_metric = float(curr_dialogue_goal_joint_correct) / curr_dialogue_goal_joint_total
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


def track_dialogue_woz(model_variables, word_vectors, dialogue_ontology, woz_dialogue, sessions):
    """
    This produces a list of belief states predicted for the given WOZ dialogue. 
    """

    prev_belief_states = {}
    belief_states = {}  # for each slot, a list of numpy arrays.

    turn_count = len(woz_dialogue)
    # print "Turn count:", turn_count

    slots_to_track = list(set(dialogue_ontology.keys()) & set(sessions.keys()))

    for slot in slots_to_track:
        belief_states[slot] = {}
        if slot != "request":
            value_count = len(dialogue_ontology[slot]) + 1
            prev_belief_states[slot] = numpy.zeros((value_count,), dtype="float32")

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

                updated_belief_state = sliding_window_over_utterance(sessions[slot], example, word_vectors,
                                                                     dialogue_ontology, mx, slot, print_mode=False)
                # updated_belief_state[updated_belief_state < 0.5] = 0
                list_of_belief_states[idx]["request"] = updated_belief_state

            else:
                updated_belief_state = sliding_window_over_utterance(sessions[slot], example, word_vectors,
                                                                     dialogue_ontology, mx, slot, print_mode=False)
                # updated_belief_state = softmax(updated_belief_state)
                # updated_belief_state = update_belief_state(prev_belief_states[slot], new_belief_state)
                prev_belief_states[slot] = updated_belief_state
                list_of_belief_states[idx][slot] = updated_belief_state

            for idx_value, value in enumerate(dialogue_ontology[slot]):
                if slot in "request":
                    current_bs[slot] = print_belief_state_woz_requestables(dialogue_ontology[slot],
                                                                           updated_belief_state, threshold=0.5)
                else:
                    current_bs[slot] = print_belief_state_woz_informable(dialogue_ontology[slot], updated_belief_state,
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


def print_belief_state_woz_informable(curr_values, distribution, threshold):
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


def print_belief_state_woz_requestables(curr_values, distribution, threshold):
    """
    Returns the top one if it is above threshold.
    """
    requested_slots = []

    # now we just print to JSON file:
    for idx, value in enumerate(curr_values):

        if distribution[idx] >= threshold:
            requested_slots.append(value)

    return requested_slots


def evaluate_model(dataset_name, sess, model_variables, data, target_slot, utterances, dialogue_ontology, \
                   positive_examples, negative_examples, print_mode=False, epoch_id=""):
    start_time = time.time()

    keep_prob, x_full, x_delex, \
    requested_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, accuracy, \
    f_score, precision, recall, num_true_positives, \
    num_positives, classified_positives, y, predictions, true_predictions, correct_prediction, \
    true_positives, train_step, update_coefficient = model_variables

    (xs_full, xs_sys_req, xs_conf_slots, xs_conf_values, xs_delex, xs_labels, xs_prev_labels) = data

    example_count = xs_full.shape[0]

    label_size = xs_labels.shape[1]

    batch_size = 16
    word_vector_size = 300
    longest_utterance_length = 40

    batch_count = int(math.ceil(float(example_count) / batch_size))

    total_accuracy = 0.0
    element_count = 0

    total_num_FP = 0.0  # FP
    total_num_TP = 0.0  # TP
    total_num_FN = 0.0  # FN -> prec = TP / (TP + FP), recall = TP / (TP + FN)
    total_num_TN = 0.0

    for idx in range(0, batch_count):

        left_range = idx * batch_size
        right_range = min((idx + 1) * batch_size, example_count)
        curr_len = right_range - left_range  # in the last batch, could be smaller than batch size

        if idx in [batch_count - 1, 0]:
            xss_full = numpy.zeros((batch_size, longest_utterance_length, word_vector_size), dtype="float32")
            xss_sys_req = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_conf_slots = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_conf_values = numpy.zeros((batch_size, word_vector_size), dtype="float32")
            xss_delex = numpy.zeros((batch_size, label_size), dtype="float32")
            xss_labels = numpy.zeros((batch_size, label_size), dtype="float32")
            xss_prev_labels = numpy.zeros((batch_size, label_size), dtype="float32")

        xss_full[0:curr_len, :, :] = xs_full[left_range:right_range, :, :]
        xss_sys_req[0:curr_len, :] = xs_sys_req[left_range:right_range, :]
        xss_conf_slots[0:curr_len, :] = xs_conf_slots[left_range:right_range, :]
        xss_conf_values[0:curr_len, :] = xs_conf_values[left_range:right_range, :]
        xss_delex[0:curr_len, :] = xs_delex[left_range:right_range, :]
        xss_labels[0:curr_len, :] = xs_labels[left_range:right_range, :]
        xss_prev_labels[0:curr_len, :] = xs_prev_labels[left_range:right_range, :]

        # ==============================================================================================

        [current_predictions, current_y, current_accuracy, update_coefficient_load] = sess.run(
            [predictions, y, accuracy, update_coefficient],
            feed_dict={x_full: xss_full, x_delex: xss_delex, \
                       requested_slots: xss_sys_req, system_act_confirm_slots: xss_conf_slots, \
                       system_act_confirm_values: xss_conf_values, y_: xss_labels, y_past_state: xss_prev_labels,
                       keep_prob: 1.0})

        #       below lines print predictions for small batches to see what is being predicted
        #        if idx == 0 or idx == batch_count - 2:
        #            #print current_y.shape, xss_labels.shape, xs_labels.shape
        #            print "\n\n", numpy.argmax(current_y, axis=1), "\n", numpy.argmax(xss_labels, axis=1), "\n==============================\n\n"

        total_accuracy += current_accuracy
        element_count += 1

    eval_accuracy = round(total_accuracy / element_count, 3)

    if print_mode:
        print
        "Epoch", epoch_id, "[Accuracy] = ", eval_accuracy, " ----- update coeff:", update_coefficient_load  # , round(end_time - start_time, 1), "seconds. ---"

    return eval_accuracy


def print_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
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


def return_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
    """
    Prints all the activated slot values for the provided predictions
    """
    predicted_values = []
    for idx, value in enumerate(slot_values):
        if distribution[idx] >= threshold:
            predicted_values += ((value, round(distribution[idx], 2)))  # , round(post_sf[idx], 2) ))

    return predicted_values


def sliding_window_over_utterance(sess, utterance, word_vectors, dialogue_ontology, model_variables, target_slot,
                                  print_mode=True):
    """
    """

    if type(model_variables) is dict:
        model_variables = model_variables[target_slot]

    list_of_outputs = test_utterance(sess, utterance, word_vectors, dialogue_ontology, model_variables, target_slot,
                                     print_mode)
    belief_state = list_of_outputs[0]

    return belief_state


def test_utterance(sess, utterances, word_vectors, dialogue_ontology, model_variables, target_slot, do_print=True):
    """
    Returns a list of belief states, to be weighted later.
    """

    potential_values = dialogue_ontology[target_slot]

    if target_slot == "request":
        value_count = len(potential_values)
    else:
        value_count = len(potential_values) + 1

    # should be a list of features for each ngram supplied.
    fv_tuples = extract_feature_vectors(utterances, word_vectors, use_asr=True)
    utterance_count = len(utterances)

    belief_state = numpy.zeros((value_count,), dtype="float32")

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

        prev_belief_state_vector = numpy.zeros((value_count,), dtype="float32")

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

    slot_values = numpy.array(slot_values)
    candidate_values = numpy.array(candidate_values)
    delexicalised_features = numpy.array(
        delexicalised_features)  # will be [batch_size, label_size, longest_utterance_length, vector_dimension]

    fv_sys_req = numpy.array(fv_sys_req)
    fv_conf_slot = numpy.array(fv_conf_slot)
    fv_conf_val = numpy.array(fv_conf_val)
    fv_full = numpy.array(fv_full)
    features_previous_state = numpy.array(features_previous_state)

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



def main():
    config_filepath = sys.argv[2]

    NBT = NeuralBeliefTracker(config_filepath)  # initialize everything

    do_training = False
    do_woz = False

    switch = sys.argv[1]
    if switch == "train":
        do_training = True
    elif switch == "woz":  # test on woz
        do_woz = True


    if do_training:
        NBT.train()
    elif do_woz:
        NBT.test_woz()
    else:   # online test
        previous_belief_state = None

        while True:

            # This means that the model must consider the three-way interaction
            # between the utterance, candidate slot-value pair and the slot value pair offered by the system.

            # two previous system acts, zero vector if none
            # Let tq and (ts, tv) be the word vectors of the arguments for the system request and confirm acts (zero vectors if none).
            # The model computes the following measures of similarity
            req_slot = input("Enter system requirement slot:")  # what price range would you like? Any
            conf_slot = input("Enter system confirm slot:") # how about Turkish food?
            conf_value = input("Enter system confirm value:") # with yes

            if req_slot not in NBT.dialogue_ontology:
                print(req_slot, "---", NBT.dialogue_ontology.keys())
                req_slot = ""
            else:
                req_slot = req_slot

            if conf_slot not in NBT.dialogue_ontology:
                conf_slot = ""
                conf_value = ""
            elif conf_value not in NBT.dialogue_ontology[conf_slot]:
                conf_slot = ""
                conf_value = ""
            else:
                conf_slot = conf_slot
                conf_value = conf_value

            utterance = input("Enter utterance for prediction:")

            if "reset" in utterance:
                previous_belief_state = None
            else:
                predictions, previous_belief_state = NBT.track_utterance(utterance, req_slot=req_slot,
                                                                         conf_slot=conf_slot, conf_value=conf_value,
                                                                         past_belief_state=previous_belief_state)
                print(json.dumps(predictions, indent=4))


if __name__ == "__main__":
    main()
