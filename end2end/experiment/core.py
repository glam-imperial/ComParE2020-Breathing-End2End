import numpy as np
import tensorflow as tf
from scipy import sparse as spsp
from scipy import signal as spsig
from scipy import stats as spst
import sklearn

from end2end.common import dict_to_struct


def get_cnn_lstm_model(audio,
                       batch_size,
                       num_layers,
                       hidden_units,
                       number_of_outputs):
    _, seq_length, num_features = audio.get_shape().as_list()
    audio_input = tf.reshape(audio, [batch_size, num_features * seq_length, 1])

    net = tf.layers.conv1d(audio_input, 64, 8, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 10, 10)

    net = tf.layers.conv1d(net, 128, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)

    net = tf.layers.conv1d(net, 256, 6, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling1d(net, 8, 8)

    net = tf.reshape(net, [batch_size, seq_length, 256])

    _, seq_length, num_features = net.get_shape().as_list()

    def _get_cell(l_no):
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True)
        return lstm

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no) for l_no in range(num_layers)], state_is_tuple=True)

    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, tf.reshape(net, (batch_size, seq_length, num_features)),
                                   dtype=tf.float32)

    if seq_length is None:
        seq_length = -1

    net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

    prediction = tf.layers.dense(net, number_of_outputs)

    prediction = tf.reshape(prediction, (batch_size, seq_length, number_of_outputs, 1))

    return prediction


def flatten_data(data, flattened_size):
    flattened_data = tf.reshape(data[:, :],
                                (-1,))
    flattened_data = tf.reshape(flattened_data,
                                (flattened_size, 1, 1, 1))
    return flattened_data


def loss_function(pred_upper_belt, true_upper_belt):
    true_upper_belt = tf.reshape(true_upper_belt, (-1, 1, 1, 1))

    support = tf.ones_like(true_upper_belt)

    return weighted_pearson_cc(pred_upper_belt, support, true_upper_belt)


def weighted_pearson_cc(pred, support, true):
    mu_x = weighted_mean(pred, support)
    mu_y = weighted_mean(true, support)

    mean_cent_prod = weighted_covariance(pred, true, mu_x, mu_y, support)
    denom = tf.sqrt(weighted_covariance(pred, pred, mu_x, mu_x, support)) +\
            tf.sqrt(weighted_covariance(true, true, mu_y, mu_y, support))

    return 1.0 - mean_cent_prod / denom


def weighted_mean(x, w):
    mu = tf.reduce_sum(tf.multiply(x, w)) / tf.reduce_sum(w)
    return mu


def weighted_covariance(x, y, mu_x, mu_y, w):
    sigma = tf.reduce_sum(tf.multiply(w, tf.multiply(x - mu_x, y - mu_y))) / tf.reduce_sum(w)
    return sigma


def get_measures_slope(items):
    measures = dict()

    pearson_upper = batch_pearson_cc_numpy(
                                                      items.upper_belt.pred,
                                           np.ones_like(
                                                                   items.upper_belt.pred),
                                           items.upper_belt.true)
    measures["pearson_upper"] = pearson_upper
    measures["pearson"] = pearson_upper

    return measures


def batch_pearson_cc_numpy(pred, support, true):

    pred = pred.reshape((-1, ))
    true = true.reshape((-1, ))

    pcc = spst.pearsonr(pred, true)[0]

    return pcc


def replace_dict_value(input_dict, old_value, new_value):
    for k, v in input_dict.items():
        if isinstance(v, str):
            if v == old_value:
                input_dict[k] = np.nan_to_num(new_value, copy=True)
    return input_dict


class RunEpoch:
    def __init__(self,
                  sess,
                  partition,
                  init_op,
                  steps_per_epoch,
                  next_element,
                  batch_size,
                  seq_length,
                  input_gaussian_noise,
                  optimizer,
                  loss,
                  pred,
                  input_feed_dict,
                  targets):
        self.sess = sess
        self.partition = partition
        self.init_op = init_op
        self.steps_per_epoch = steps_per_epoch
        self.next_element = next_element
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.optimizer = optimizer
        self.loss = loss
        self.pred = pred
        self.input_gaussian_noise = input_gaussian_noise
        self.input_feed_dict = input_feed_dict
        self.targets = targets

        self.number_of_targets = len(self.targets)

    def run_epoch(self):
        batch_size_sum = 0

        # Initialize an iterator over the dataset split.
        self.sess.run(self.init_op)

        # Store variable sequence.
        stored_variables = dict()
        for target in self.targets:
            stored_variables[target] = dict()
            if self.partition in ["train", "devel"]:
                stored_variables[target]["true"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                             self.seq_length),
                                                            dtype=np.float32)

            stored_variables[target]["pred"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                         self.seq_length),
                                                        dtype=np.float32)
        stored_variables["loss"] = None

        # Run epoch.
        subject_to_id = dict()
        for step in range(self.steps_per_epoch):
            batch_tuple = self.sess.run(self.next_element)
            sample_id = batch_tuple["sample_id"]
            recording_id = batch_tuple["recording_id"]
            audio = batch_tuple["raw_audio"]
            if self.partition in ["train", "devel"]:
                upper_belt = batch_tuple["upper_belt"]

            batch_size_sum += audio.shape[0]

            subject_to_id[recording_id[0, 0][0]] = step

            seq_pos_start = step * self.batch_size
            seq_pos_end = seq_pos_start + audio.shape[0]

            # Augment data.
            jitter = np.random.normal(scale=self.input_gaussian_noise,
                                      size=audio.shape)
            audio_plus_jitter = audio + jitter

            feed_dict = {k: v for k, v in self.input_feed_dict.items()}
            feed_dict = replace_dict_value(feed_dict, "batch_size", audio.shape[0])
            feed_dict = replace_dict_value(feed_dict, "audio", audio_plus_jitter)
            if self.partition in ["train", "devel"]:
                feed_dict = replace_dict_value(feed_dict, "upper_belt", upper_belt)

            out_tf = list()
            out_tf.append(self.pred)
            optimizer_index = None
            loss_index = None
            if self.optimizer is not None:
                out_tf.append(self.optimizer)
                optimizer_index = len(out_tf) - 1
            if self.loss is not None:
                out_tf.append(self.loss)
                loss_index = len(out_tf) - 1

            out_np = self.sess.run(out_tf,
                              feed_dict=feed_dict)

            if self.partition in ["train", "devel"]:
                stored_variables["upper_belt"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(upper_belt)
            stored_variables["upper_belt"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0].reshape((self.batch_size,
                                                                                                      self.seq_length))

            if self.loss is not None:
                stored_variables["loss"] = out_np[loss_index]

        for target in self.targets:
            if self.partition in ["train", "devel"]:
                stored_variables[target]["true"] = stored_variables[target]["true"][:batch_size_sum, :]
            stored_variables[target]["pred"] = stored_variables[target]["pred"][:batch_size_sum, :]
            stored_variables[target] = dict_to_struct(stored_variables[target])

        stored_variables = dict_to_struct(stored_variables)

        return stored_variables, subject_to_id


def make_label_file(path, data):
    with open(path, "w") as fp:
        for speaker in range(data.shape[0]):

            if speaker < 10:
                speaker_text = "0" + repr(speaker)
            else:
                speaker_text = repr(speaker)

            t = 0.00
            for j in range(data.shape[1]):

                fp.write("test_" + speaker_text + ".wav" + "," + '%.2f' % t + "," + '%.5f' % data[speaker, j] + "\n")

                t += 0.04
