import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import *


class NBT_model(nn.Module):
    def __init__(self, vector_dimension, label_count, slot_ids, value_ids, use_delex_features=False,
                 use_softmax=True, value_specific_decoder=False, learn_belief_state_update=True,
                 embedding=None, float_tensor=torch.FloatTensor, long_tensor=torch.LongTensor, target_slot=None,
                 value_list=None, longest_utterance_length=40,
                 num_filters=300, drop_out=0.5, lr=1e-4, device=torch.device("cpu")):
        super(NBT_model, self).__init__()

        self.slot_emb = embedding(slot_ids)

        print("self.slot_emb.shape", self.slot_emb.shape)

        self.value_emb = embedding(value_ids)
        # self.slot_value_pair_emb = torch.cat((self.slot_emb, self.value_emb), dim=1)  # cs+cv
        # self.w_candidates = nn.Linear(vector_dimension * 2, vector_dimension, bias=True)  # equation (7) in paper 1
        self.slot_value_pair_emb = self.slot_emb + self.value_emb  # TODO: double check cs+cv in equation (7)
        self.w_candidates = nn.Linear(vector_dimension, vector_dimension, bias=True)  # equation (7) in paper 1

        if device == torch.device("cuda:0"):
            self.w_candidates = self.w_candidates.cuda()

        self.target_slot = target_slot
        self.value_list = value_list
        self.filter_sizes = [1, 2, 3]
        self.num_filters = 300
        self.drop_out = drop_out
        self.hidden_utterance_size = self.num_filters  # * len(filter_sizes)
        self.vector_dimension = vector_dimension
        self.longest_utterance_length = longest_utterance_length
        self.use_softmax = use_softmax and target_slot != 'request'
        assert (use_softmax ^ (
                target_slot == "request"))  # if request, choose 1 of the 7; if not, has the option of value=None

        self.use_delex_features = use_delex_features
        self.hidden_units = 100  # before equation (11)
        self.filter_sizes = [1, 2, 3]
        self.conv_filters = [None, None, None]
        self.dnn_filters = [None, None, None]
        self.lstm = nn.LSTM(300, self.hidden_units, batch_first=True, bidirectional=False)

        # self.mlp_post_lstm = []
        #
        # for i, n in enumerate(range(len(self.value_list))):
        #     self.mlp_post_lstm += [
        #         nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
        #                       nn.Linear(50, 1, bias=True), nn.Sigmoid())]

        self.mlp_post_lstm0 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm1 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm2 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm3 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm4 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm5 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))
        self.mlp_post_lstm6 = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, 2, bias=True))

        self.mlp_post_lstmn = nn.Sequential(nn.Linear(self.hidden_units, 50, bias=True), nn.ReLU(),
                                            nn.Linear(50, len(self.value_list) + 1, bias=True))

        self.float_tensor = float_tensor
        self.long_tensor = long_tensor

        for i, n in enumerate(self.filter_sizes):

            self.conv_filters[i] = nn.Conv1d(self.vector_dimension, self.num_filters, n, bias=True)
            self.dnn_filters[i] = nn.Sequential(
                nn.Linear(n, 1, bias=True), nn.Sigmoid())

            if device == torch.device("cuda:0"):
                self.conv_filters[i] = self.conv_filters[i].cuda()
                self.dnn_filters[i] = self.dnn_filters[i].cuda()

        # Equation 11

        self.w_hidden_layer_for_d = nn.Sequential(nn.Linear(self.vector_dimension, self.hidden_units, bias=True),
                                                  nn.Sigmoid())

        self.w_joint_presoftmax = nn.Sequential(nn.Linear(self.hidden_units, 1, bias=True), nn.Sigmoid())

        self.w_hidden_layer_for_mr = nn.Sequential(
            nn.Linear(self.vector_dimension, self.hidden_units, bias=True), nn.Sigmoid())
        self.w_hidden_layer_for_mc = nn.Sequential(
            nn.Linear(self.vector_dimension, self.hidden_units, bias=True), nn.Sigmoid())

        if device == torch.device("cuda:0"):
            self.w_hidden_layer_for_d = self.w_hidden_layer_for_d.cuda()
            self.w_joint_presoftmax = self.w_joint_presoftmax.cuda()
            self.w_hidden_layer_for_mr = self.w_hidden_layer_for_mr.cuda()
            self.w_hidden_layer_for_mc = self.w_hidden_layer_for_mc.cuda()

        self.combine_coefficient = 0.5
        self.device = device

    def define_DNN_model(self, utterance_representation_full):

        """
        Better source for defining the CNN model.


        utterance_representations_full: shape minibatch_size * seqLen * emb_dim -> reshape to minibatch_size * emb_dim * seqLen

        """
        inputTensor = utterance_representation_full.permute(0, 2, 1)  # 512 * 300 * 40
        # print(input)
        #
        # input("Input: whether self final vectores are zero " + str(inputTensor[3, -5:, -5:]))

        for i, n in enumerate(self.filter_sizes):
            r_n = inputTensor[:, :, :n]
            for start in range(1, self.longest_utterance_length - n):
                end = start + n
                r_n += inputTensor[:, :, start:end]

            # input("Input: rn in equation (2), 512 * 300 * n " + str(r_n.shape))

            r_n = self.dnn_filters[i](r_n)
            # input("Input: rn in equation (3), 512 * 300 * 1 " + str(r_n.shape))

            if i == 0:
                final_utterance = r_n
            else:
                final_utterance += r_n

        return final_utterance

    def define_CNN_model(self, utterance_representations_full):
        """
        Better source for defining the CNN model.


        utterance_representations_full: shape minibatch_size * seqLen * emb_dim -> reshape to minibatch_size * emb_dim * seqLen

        """
        inputTensor = utterance_representations_full.permute(0, 2, 1)
        # print("permuted utterance_representations_full", input.shape)  # (minibatch_size, num_filter, seqLen-n+1)
        output = [None] * len(self.filter_sizes)

        for i, n in enumerate(self.filter_sizes):
            print("input.shape", inputTensor.shape)
            output[i] = F.relu(self.conv_filters[i](inputTensor))
            print("output[i].shape " + str(output[i].shape))
            print("input.shape[2]-n+1 " + str(inputTensor.shape[2] - n + 1))

            output[i] = F.max_pool1d(output[i], inputTensor.shape[2] - n + 1)
            print("output[i].shape " + str(output[i].shape))

        final_utterance = output[0]

        for i in range(1, len(self.filter_sizes)):
            # print(output[i].shape)
            final_utterance += output[i]

        # print(final_utterance.shape)

        return final_utterance

    def define_LSTM_model(self, utterance_representations_full, utterance_lens):

        """
        Better source for defining the CNN model.


        utterance_representations_full: shape minibatch_size * seqLen * emb_dim -> reshape to minibatch_size * emb_dim * seqLen

        """

        inputTensor = utterance_representations_full[:, :max(utterance_lens), :]

        packedInputTensor = pack_padded_sequence(inputTensor, lengths=self.long_tensor(utterance_lens),
                                                 batch_first=True)

        outputTensor, (h_n, c_n) = self.lstm(packedInputTensor)  # outputTensor(40,512,100) h_n shape of 2*512*100

        # lastTimestamp=outputTensor[]

        # unpackedOutputTensor,_ = pad_packed_sequence(outputTensor, batch_first=True)
        # print(utterance_lens - np.ones(len(utterance_lens)))
        # sentence_representation_full = unpackedOutputTensor[:, utterance_lens - np.ones(len(utterance_lens)), :]
        # sentence_representation_full_fake = unpackedOutputTensor[:, -1, :]  # should have only first row as 1
        #
        # print("inside define_LSTM_model")

        return h_n
        # return utterance_representations_full

    def forward(self, batch_data, keep_prob=0.5):
        """

        :param packed_data: utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots, system_act_confirm_values, y_, y_past_state, keep_prob
        where y_ size is (None, label_size)
        :return:
        """
        # (utterance_representations_full, utterance_representation_delex, system_act_slots, system_act_confirm_slots,
        #  system_act_confirm_values, y_, y_past_state, keep_prob) = batch_data

        (utterance_representations_full, system_act_slots, system_act_confirm_slots, system_act_confirm_values,
         utterance_representation_delex, y_, y_past_state, utterance_lens) = batch_data

        # print("within model forward "+str(utterance_representations_full))

        # candidates_transform = F.sigmoid(self.w_candidates(
        #     self.slot_value_pair_emb))  # equation (7) in paper 1, candidates_transform has size len(value_list) * vector_dimension

        candidates_transform = F.sigmoid(self.w_candidates(
            self.slot_value_pair_emb))  # equation (7) in paper 1, candidates_transform has size len(value_list) * vector_dimension

        # input("candidates_transform.shape"+str(candidates_transform.shape))

        # print("candidates_transofrm shape " + str(candidates_transform.shape))
        # print("current target slot is "+self.target_slot)
        # input("c_size "+str(c.shape))

        # TODO: cross experiment one of the three variants
        # final_utterance_representation = self.define_CNN_model(utterance_representations_full)
        # final_utterance_representation = self.define_DNN_model(utterance_representations_full)
        final_utterance_representation = self.define_LSTM_model(utterance_representations_full, utterance_lens)

        final_utterance_representation = final_utterance_representation.squeeze(0)

        # for i in range(len(self.value_list)):

        # y_list[:, 0] = self.mlp_post_lstm0(final_utterance_representation).squeeze(0)
        # y_list[:, 1] = self.mlp_post_lstm1(final_utterance_representation).squeeze(0)
        # y_list[:, 2] = self.mlp_post_lstm2(final_utterance_representation).squeeze(0)
        # y_list[:, 3] = self.mlp_post_lstm3(final_utterance_representation).squeeze(0)
        # y_list[:, 4] = self.mlp_post_lstm4(final_utterance_representation).squeeze(0)
        # y_list[:, 5] = self.mlp_post_lstm5(final_utterance_representation).squeeze(0)
        # y_list[:, 6] = self.mlp_post_lstm6(final_utterance_representation).squeeze(0)
        if self.target_slot == "request":

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred0 = self.mlp_post_lstm0(final_utterance_representation)
            loss = loss_func(pred0, y_[:, 0].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred1 = self.mlp_post_lstm1(final_utterance_representation)
            loss += loss_func(pred1, y_[:, 1].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred2 = self.mlp_post_lstm2(final_utterance_representation)
            loss += loss_func(pred2, y_[:, 2].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred3 = self.mlp_post_lstm3(final_utterance_representation)
            loss += loss_func(pred3, y_[:, 3].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred4 = self.mlp_post_lstm4(final_utterance_representation)
            loss += loss_func(pred4, y_[:, 4].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred5 = self.mlp_post_lstm5(final_utterance_representation)
            loss += loss_func(pred5, y_[:, 5].long())

            loss_func = nn.CrossEntropyLoss(
                weight=self.float_tensor([y_[:, 0].long().sum(), (1 - y_[:, 0].long()).sum()]))
            pred6 = self.mlp_post_lstm6(final_utterance_representation)
            loss += loss_func(pred6, y_[:, 6].long())

            # prediction_i = self.mlp_post_lstm[i](final_utterance_representation)

            # y_list += [prediction_i.squeeze(0).squeeze(1)]

            # return torch.stack(y_list).permute(1, 0)
            # return y_list.permute(1, 0)

            softmax_layer = nn.Softmax(dim=1)
            prediction = torch.stack(
                [softmax_layer(pred0)[:, 1], softmax_layer(pred1)[:, 1], softmax_layer(pred2)[:, 1],
                 softmax_layer(pred3)[:, 1], softmax_layer(pred4)[:, 1], softmax_layer(pred5)[:, 1],
                 softmax_layer(pred6)[:, 1]]).permute(1,
                                                      0)
        else:
            weight = self.float_tensor([(1 / ((y_ == i).sum().float()) if (y_ == i).sum() != 0 else 0) for i in
                                        range(len(self.value_list) + 1)])
            loss_func = nn.CrossEntropyLoss(weight=weight)
            pred = self.mlp_post_lstmn(final_utterance_representation)
            loss = loss_func(pred, y_.long())
            softmax_layer = nn.Softmax(dim=1)
            prediction = softmax_layer(pred)

        return prediction, loss

        #
        # list_of_value_contributions = []
        #
        # # get interaction of utterance with each value, element-wise multiply, Equation (8)
        # for batch_idx in range(final_utterance_representation.shape[0]):
        #     repeat_utterance = final_utterance_representation[batch_idx].squeeze(1).repeat(
        #         [candidates_transform.shape[0], 1])
        #
        #     list_of_value_contributions.append(torch.addcmul(
        #         self.float_tensor(np.zeros([candidates_transform.shape[0], repeat_utterance.shape[1]])),
        #         candidates_transform,
        #         repeat_utterance))  # candidates_transform.mul(final_utterance_representation[batch_idx]))
        #
        # # print("list of contributions[0] shape", list_of_value_contributions[0].shape)
        # list_of_value_contributions = torch.stack(
        #     list_of_value_contributions)  # minibatch_size * label_size * vector_dimension
        #
        # # y_presoftmax_1 = torch.reshape(self.w_joint_presoftmax(self.w_joint_hidden_layer(list_of_value_contributions)),
        # #                                [-1, list_of_value_contributions.shape[1]])
        #
        # y_presoftmax_1 = self.w_joint_presoftmax(
        #     F.dropout(self.w_hidden_layer_for_d(list_of_value_contributions), p=self.drop_out, training=self.training))
        #
        # # input(self.slot_emb)
        # m_r = torch.matmul(self.slot_emb.matmul(system_act_slots.t()),
        #                    final_utterance_representation.squeeze(2))
        #
        # y_presoftmax_2 = self.w_joint_presoftmax(
        #     F.dropout(self.w_hidden_layer_for_mr(m_r), p=self.drop_out, training=self.training))
        #
        # m_c = torch.matmul(
        #     torch.addcmul(self.float_tensor(np.zeros([self.slot_emb.shape[0], system_act_confirm_slots.shape[0]])),
        #                   self.slot_emb.matmul(system_act_confirm_slots.t()),
        #                   self.value_emb.matmul(system_act_confirm_values.t())),
        #     final_utterance_representation.squeeze(2))
        # y_presoftmax_3 = self.w_joint_presoftmax(
        #     F.dropout(self.w_hidden_layer_for_mc(m_c), p=self.drop_out, training=self.training))
        #
        # # Equation 11 again, lol
        # y_presoftmax = y_presoftmax_1 + y_presoftmax_2 + y_presoftmax_3
        #
        # # TODO: 1. modify loss into for-looping multiple label; 2. making sense of the y_combine interpolation
        #
        # # if self.use_softmax:
        # #     append_zeros_none = torch.Tensor(np.zeros([y_presoftmax_1.shape[0], 1, y_presoftmax_1.shape[2]]))
        # #     y_presoftmax = torch.cat((y_presoftmax, append_zeros_none), dim=1)
        #
        # if self.use_delex_features:
        #     y_presoftmax = y_presoftmax + utterance_representation_delex
        #
        # if self.use_softmax:
        #     # print("Tensor of size (mini_batch_size,label_count)?")
        #     # input(y_past_state.shape)
        #     # input(y_presoftmax.shape)
        #     y = self.combine_coefficient * y_presoftmax.squeeze(2) + (1 - self.combine_coefficient) * y_past_state
        #     # y = F.softmax(y_combine, dim=1)
        #     print("predicting non-request prior-softmax, loss is CrossEntropy")
        #
        # else:
        #     y = F.sigmoid(y_presoftmax).squeeze(2)  # comparative max is okay?
        #     print("predicting request with sigmoid, loss is L2-MSE")
        #
        # print("f_pred", y)
        #
        # return y

    def eval_model(self, val_data):

        self.eval()
        # input("setting to eval mode " + str(self.training))

        (val_xs_full, val_sys_req, val_sys_conf_slots, val_sys_conf_values,
         val_delex, val_ys, val_ys_prev, val_xs_lens) = val_data

        print("val_xs_full shape  getting forwarded" + str(val_xs_full.shape))

        f_pred, loss = self.forward(val_data)

        print("forward finished")

        print("predictions shape ", f_pred.shape)  # batch_size * label_count
        print("val_ys shape ", val_ys.shape)

        if self.device == torch.device("cuda:0"):
            f_pred = f_pred.cpu()
            val_ys = val_ys.cpu()

        if self.use_softmax:
            predictions = f_pred.argmax(1)

            correct_prediction = (predictions.long() == val_ys.long()).float()
            accuracy = np.asscalar(correct_prediction.mean().data.numpy())

            # predictions_one_hot = self.float_tensor(np.zeros(f_pred.shape)).scatter_(1, predictions.unsqueeze(1).long(), 1)

            # true_predictions = val_ys.float()
            # true_predictions_one_hot = self.float_tensor(np.zeros(f_pred.shape)).scatter_(1, true_predictions.unsqueeze(
            #     1).long(), 1)
            #
            # correct_prediction = (predictions.long() == true_predictions.long()).float()
            # accuracy = correct_prediction.mean()
            #
            # precision = 0.0
            # recall = 0.0
            #
            # for ind in range(f_pred.shape[1]):
            #     num_positives = true_predictions_one_hot[ind].sum()
            #     classified_positives = predictions_one_hot[ind].sum()
            #     true_positives = true_predictions_one_hot[ind] * predictions_one_hot[ind]
            #     num_true_positives = true_positives.sum()
            #     precision += (0 if np.asscalar(classified_positives.data.numpy()) == 0 else np.asscalar(
            #         (1.0 * num_true_positives / classified_positives).data.numpy()))
            #     recall += (0 if np.asscalar(num_positives.data.numpy()) == 0 else np.asscalar(
            #         (1.0 * num_true_positives / num_positives).data.numpy()))
            #
            # precision = precision / f_pred.shape[1]
            # recall = recall / f_pred.shape[1]
            # f_score = torch.Tensor([0]) if (recall + precision) == 0 else torch.Tensor(
            #     [(2 * recall * precision) / (recall + precision)])

        else:
            predictions = f_pred.round()
            print("validation: rounded predictions ", predictions)
            true_predictions = val_ys.float()
            print("validation: val_ys", val_ys)

            correct_prediction = (predictions.long() == true_predictions.long()).float()
            num_positives = true_predictions.sum()
            print("num_positives", num_positives)
            classified_positives = predictions.sum()
            true_positives = (predictions * true_predictions)
            num_true_positives = (true_positives).sum()
            recall = num_true_positives / num_positives
            precision = num_true_positives / classified_positives
            f_score = torch.Tensor([0]) if np.asscalar((recall + precision).data.numpy()) == 0 else (
                                                                                                            2 * recall * precision) / (
                                                                                                            recall + precision)
            accuracy = np.asscalar(correct_prediction.mean().data.numpy())

        return accuracy, "accuracy"
