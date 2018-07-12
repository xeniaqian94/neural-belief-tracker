import math
import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class NBT_model(nn.Module):

    def __init__(self, vector_dimension, label_count, slot_ids, value_ids, use_delex_features=False,
                 use_softmax=True, value_specific_decoder=False, learn_belief_state_update=True,
                 embedding=None, dtype=torch.float, device=torch.device("cpu"),
                 tensor_type=torch.FloatTensor, target_slot=None, value_list=None, longest_utterance_length=40,
                 num_filters=300, drop_out=0.5, lr=1e-4):
        super(NBT_model, self).__init__()

        self.slot_emb = embedding(slot_ids)
        self.value_emb = embedding(value_ids)
        self.slot_value_pair_emb = torch.cat((self.slot_emb, self.value_emb), dim=1)  # cs+cv

        self.w_candidates = nn.Linear(vector_dimension * 2, vector_dimension, bias=True)  # equation (7) in paper 1

        self.target_slot = target_slot
        self.value_list = value_list
        self.filter_sizes = [1, 2, 3]
        self.num_filters = 300
        self.hidden_utterance_size = self.num_filters  # * len(filter_sizes)
        self.vector_dimension = vector_dimension
        self.longest_utterance_length = longest_utterance_length
        self.use_softmax = use_softmax and target_slot != 'request'
        assert (use_softmax ^ (
                target_slot == "request"))  # if request, choose 1 of the 7; if not, has the option of value=None

        self.use_delex_features = use_delex_features
        self.filter_sizes = [1, 2, 3]
        self.conv_filters = [None, None, None]
        self.hidden_units = 100  # before equation (11)

        for i, n in enumerate(self.filter_sizes):
            self.conv_filters[i] = nn.Conv1d(self.vector_dimension, self.num_filters, n, bias=True)

        # Equation 11
        self.w_hidden_layer_for_d = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                  nn.Linear(self.vector_dimension, self.hidden_units, bias=True))
        # F.sigmoid(nn.Linear(self.vector_dimension, self.hidden_units, bias=True)), p=drop_out)
        self.w_joint_presoftmax = nn.Sequential(nn.Sigmoid(), nn.Linear(self.hidden_units, 1, bias=True))

        self.w_hidden_layer_for_mr = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                   nn.Linear(self.vector_dimension, self.hidden_units, bias=True))
        self.w_hidden_layer_for_mc = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                   nn.Linear(self.vector_dimension, self.hidden_units, bias=True))

        self.combine_coefficient = 0.5

    def define_CNN_model(self, utterance_representations_full, num_filters=300, vector_dimension=300,
                         longest_utterance_length=40):
        """
        Better code for defining the CNN model.


        utterance_representations_full: shape minibatch_size * seqLen * emb_dim -> reshape to minibatch_size * emb_dim * seqLen

        """
        input = utterance_representations_full.permute(0, 2, 1)
        # print("permuted utterance_representations_full", input.shape)  # (minibatch_size, num_filter, seqLen-n+1)
        output = [None] * len(self.filter_sizes)

        for i, n in enumerate(self.filter_sizes):
            output[i] = F.relu(self.conv_filters[i](input))
            # print(output[i].shape)
            output[i] = F.max_pool1d(output[i], input.shape[2] - n + 1)

        final_utterance = output[0]

        for i in range(1, len(self.filter_sizes)):
            # print(output[i].shape)
            final_utterance += output[i]

        # print(final_utterance.shape)

        return final_utterance

    def forward(self, batch_data, keep_prob=0.5):
        """

        :param packed_data: utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, keep_prob
        where y_ size is (None, label_size)
        :return:
        """
        # (utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots,
        #  system_act_confirm_values, y_, y_past_state, keep_prob) = batch_data

        (utterance_representations_full, system_act_slots, system_act_confirm_slots, system_act_confirm_values,
         utterance_representation_delex, y_, y_past_state) = batch_data

        # print("within model forward "+str(utterance_representations_full))

        candidates_transform = F.sigmoid(self.w_candidates(
            self.slot_value_pair_emb))  # equation (7) in paper 1, candidates_transform has size len(value_list) * vector_dimension
        # print("candidates_transofrm shape " + str(candidates_transform.shape))
        # print("current target slot is "+self.target_slot)
        # input("c_size "+str(c.shape))

        # TODO 1 after dragon-boat: change CNN filtering -> h_utterance_represetnation code logic in PyTorch
        final_utterance_representation = self.define_CNN_model(utterance_representations_full)

        # Next, multiply candidates [label_size, vector_dimension] each with the uttereance representations [None, vector_dimension], to get [None, label_size, vector_dimension]
        # or utterance [None, vector_dimension] X [vector_dimension, label_size] to get [None, label_size]
        # h_utterance_representation_candidate_interaction = tf.Variable(tf.zeros([None, label_size, vector_dimension]))

        list_of_value_contributions = []

        # get interaction of utterance with each value, element-wise multiply, Equation (8)
        for batch_idx in range(final_utterance_representation.shape[0]):
            repeat_utterance = final_utterance_representation[batch_idx].squeeze(1).repeat(
                [candidates_transform.shape[0], 1])

            list_of_value_contributions.append(torch.addcmul(
                torch.zeros(candidates_transform.shape[0], repeat_utterance.shape[1]),
                candidates_transform,
                repeat_utterance))  # candidates_transform.mul(final_utterance_representation[batch_idx]))

        # print("list of contributions[0] shape", list_of_value_contributions[0].shape)
        list_of_value_contributions = torch.stack(
            list_of_value_contributions)  # minibatch_size * label_size * vector_dimension

        # y_presoftmax_1 = torch.reshape(self.w_joint_presoftmax(self.w_joint_hidden_layer(list_of_value_contributions)),
        #                                [-1, list_of_value_contributions.shape[1]])

        y_presoftmax_1 = self.w_joint_presoftmax(self.w_hidden_layer_for_d(list_of_value_contributions))

        # print(y_presoftmax_1.shape, "hopefully is minibatch_size * label_size * vector_dimension")

        # =======================
        # Calculating sysreq and confirm contribution

        # Equation (9)
        # system_act_slots = (minibatch_size, vector_dimension)

        # print(self.slot_emb.shape)  # 7*300
        # print("system_act_slots", system_act_slots.shape)  # 7*300
        # print("system_act_confirm_slots", system_act_confirm_slots.shape)
        # print(final_utterance_representation.shape)  # 512 * 300

        m_r = torch.matmul(self.slot_emb.matmul(system_act_slots.t()),
                           final_utterance_representation.squeeze(2))

        y_presoftmax_2 = self.w_joint_presoftmax(self.w_hidden_layer_for_mr(m_r))

        m_c = torch.matmul(torch.addcmul(torch.zeros(self.slot_emb.shape[0], system_act_confirm_slots.shape[0]),
                                         self.slot_emb.matmul(system_act_confirm_slots.t()),
                                         self.value_emb.matmul(system_act_confirm_values.t())),
                           final_utterance_representation.squeeze(2))
        y_presoftmax_3 = self.w_joint_presoftmax(self.w_hidden_layer_for_mc(m_c))

        # Equation 11 again, lol
        y_presoftmax = y_presoftmax_1 + y_presoftmax_2 + y_presoftmax_3

        # TODO: 1. modify loss into for-looping multiple label; 2. making sense of the y_combine interpolation

        # if self.use_softmax:
        #     append_zeros_none = torch.Tensor(np.zeros([y_presoftmax_1.shape[0], 1, y_presoftmax_1.shape[2]]))
        #     y_presoftmax = torch.cat((y_presoftmax, append_zeros_none), dim=1)

        if self.use_delex_features:
            y_presoftmax = y_presoftmax + utterance_representation_delex

        if self.use_softmax:
            # print("Tensor of size (mini_batch_size,label_count)?")
            # input(y_past_state.shape)
            # input(y_presoftmax.shape)
            y = self.combine_coefficient * y_presoftmax.squeeze(2) + (1 - self.combine_coefficient) * y_past_state
            # y = F.softmax(y_combine, dim=1)
            print("predicting non-request prior-softmax, loss is CrossEntropy")

        else:
            y = F.sigmoid(y_presoftmax)  # comparative max is okay?
            print("predicting request with sigmoid, loss is L2-MSE")

        # y = y.squeeze(2)

        return y

    def eval_model(self, val_data):

        (val_xs_full, val_sys_req, val_sys_conf_slots, val_sys_conf_values,
         val_delex, val_ys, val_ys_prev) = val_data

        f_pred = self.forward(val_data)

        if self.use_softmax:
            predictions = f_pred.argmax(1)
        else:
            predictions = f_pred.round()

        # input(predictions)
        true_predictions = val_ys.float()
        predictions=predictions.float()
        # input(true_predictions)

        correct_prediction = (predictions == true_predictions).float()

        # input(correct_prediction)
        # input(predictions.sum())
        accuracy = correct_prediction.mean()
        num_positives = true_predictions.sum()
        classified_positives = predictions.sum()
        # will have ones in all places where both are predicting positives
        true_positives = predictions * true_predictions
        # if indicators for positive of both are 1, then it is positive.
        num_true_positives = true_positives.sum()

        recall = num_true_positives / num_positives
        precision = num_true_positives / classified_positives
        f_score = (2 * recall * precision) / (recall + precision)

        return np.asscalar(accuracy.data.numpy())
