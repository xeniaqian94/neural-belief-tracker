import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class NBT_model(nn.Module):

    def __init__(self, vector_dimension, label_count, slot_ids, value_ids, use_delex_features=False,
                 use_softmax=True, value_specific_decoder=False, learn_belief_state_update=True,
                 embedding=None, dtype=torch.float, device=torch.device("cpu"),
                 tensor_type=torch.FloatTensor, target_slot=None, value_list=None, longest_utterance_length=40,
                 num_filters=300, drop_out=0.5):
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

        self.filter_sizes = [1, 2, 3]
        self.conv_filters = [None, None, None]
        self.hidden_units = 100

        for i, n in enumerate(self.filter_sizes):
            self.conv_filters[i] = nn.Conv1d(self.vector_dimension, self.num_filters, n, bias=True)

        # Equation 7 and 8
        self.w_joint_hidden_layer = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                  nn.Linear(self.vector_dimension, self.hidden_units, bias=True))
        # F.sigmoid(nn.Linear(self.vector_dimension, self.hidden_units, bias=True)), p=drop_out)
        self.w_joint_presoftmax = nn.Sequential(nn.Sigmoid(), nn.Linear(self.hidden_units, 1, bias=True))

        # Equation 9
        self.sysreq_w_hidden_layer = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                   nn.Linear(self.vector_dimension, self.hidden_units, bias=True))
        self.sysreq_w_presoftmax = nn.Sequential(nn.Sigmoid(), nn.Linear(self.hidden_units, 1, bias=True))

        # Equation 10

        self.syscfm_w_hidden_layer = nn.Sequential(nn.Dropout(p=drop_out), nn.Sigmoid(),
                                                   nn.Linear(self.vector_dimension, self.hidden_units, bias=True))
        self.syscfm_w_presoftmax = nn.Sequential(nn.Sigmoid(), nn.Linear(self.hidden_units, 1, bias=True))

        self.update_coefficient = 0.5

    def define_CNN_model(self, utterance_representations_full, num_filters=300, vector_dimension=300,
                         longest_utterance_length=40):
        """
        Better code for defining the CNN model.


        utterance_representations_full: shape minibatch_size * seqLen * emb_dim -> reshape to minibatch_size * emb_dim * seqLen

        """
        input = utterance_representations_full.permute(0, 2, 1)
        print("permuted utterance_representations_full", input.shape)
        output = [None] * self.filter_sizes

        for i, n in enumerate(self.filter_sizes):
            # print("processing with kernel size " + str(i))
            output[i] = F.relu(self.conv_filters[i](input))
            output[i] = F.max_pool1d(output[i].permute(0, 2, 1), self.num_filters).permute(0, 2, 1)

        final_utterance = output[0]

        for i in range(1, self.filter_sizes):
            final_utterance += output[i]

        return final_utterance

    def forward(self, packed_data):
        """

        :param packed_data: utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, keep_prob
        where y_ size is (None, label_size)
        :return:
        """
        (utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots,
         system_act_confirm_values, y_, y_past_state, keep_prob) = packed_data

        # print("within model forward "+str(utterance_representations_full))

        candidates_transform = F.sigmoid(self.w_candidates(self.slot_value_pair_emb))  # equation (7) in paper 1

        # print("current target slot is "+self.target_slot)
        # input("c_size "+str(c.shape))

        # TODO 1 after dragon-boat: change CNN filtering -> h_utterance_represetnation code logic in PyTorch
        final_utterance_representation = self.define_CNN_model(utterance_representations_full)

        print("is this utterance shape = (minibatch_size, vector_dim) ? " + str(final_utterance_representation.shape))

        # Next, multiply candidates [label_size, vector_dimension] each with the uttereance representations [None, vector_dimension], to get [None, label_size, vector_dimension]
        # or utterance [None, vector_dimension] X [vector_dimension, label_size] to get [None, label_size]
        # h_utterance_representation_candidate_interaction = tf.Variable(tf.zeros([None, label_size, vector_dimension]))

        list_of_value_contributions = []

        # get interaction of utterance with each value, element-wise multiply, equation (8)
        for batch_idx in range(final_utterance_representation.shape[0]):
            list_of_value_contributions.append(candidates_transform.mul(final_utterance_representation[batch_idx]))
        list_of_value_contributions = torch.stack(
            list_of_value_contributions)  # minibatch_size * label_size * vector_dimension

        y_presoftmax_1 = torch.reshape(self.w_joint_presoftmax(self.w_joint_hidden_layer(list_of_value_contributions)),
                                       [-1, list_of_value_contributions.shape[1]])

        # =======================
        # Calculating sysreq and confirm contribution

        # system_act_slots = (minibatch_size, vector_dimension) # Equation (9)
        m_r = torch.matmul(self.slot_emb.matmul(system_act_slots),
                           final_utterance_representation)

        y_presoftmax_2 = self.sysreq_w_presoftmax(self.sysreq_w_hidden_layer(m_r))

        m_c = torch.matmul(torch.matmul(self.slot_emb.matmul(system_act_confirm_slots),
                                        self.value_emb.matmul(system_act_confirm_values)),
                           final_utterance_representation)
        y_presoftmax_3 = self.syscfm_w_presoftmax(self.syscfm_w_hidden_layer(m_c)), [-1,
                                                                                     list_of_value_contributions.shape[
                                                                                         1]]

        if self.use_softmax:
            append_zeros_none = torch.Tensor(np.zeros([y_presoftmax_1.shape[0], 1]))
            y_presoftmax_1 = torch.concat([y_presoftmax_1, append_zeros_none], 1)
            y_presoftmax_2 = torch.concat([y_presoftmax_2, append_zeros_none], 1)
            y_presoftmax_3 = torch.concat([y_presoftmax_3, append_zeros_none], 1)
            y_presoftmax = y_presoftmax_1 + y_presoftmax_2 + y_presoftmax_3

        if self.use_delex_features:
            y_presoftmax = y_presoftmax + utterance_representation_delex

        if self.use_softmax:
            y_combine = self.update_coefficient * y_presoftmax + (1 - self.update_coefficient) * y_past_state
            y = nn.softmax(y_combine, dim=1)
        else:
            y = nn.sigmoid(y_presoftmax)

        return y
