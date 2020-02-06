import os

import tensorflow as tf

import end2end.experiment.core as core
import end2end.data_read.data_provider as data_provider
from end2end.common import dict_to_struct, make_dirs_safe


def train(configuration):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(configuration.GPU)

    ####################################################################################################################
    # Interpret configuration arguments.
    ####################################################################################################################
    train_steps_per_epoch = (configuration.full_seq_length * configuration.train_size) //\
                            (configuration.train_seq_length * configuration.train_batch_size)
    devel_steps_per_epoch = (configuration.full_seq_length * configuration.devel_size) //\
                            (configuration.full_seq_length * configuration.devel_batch_size)
    test_steps_per_epoch = (configuration.full_seq_length * configuration.test_size) // \
                           (configuration.full_seq_length * configuration.test_batch_size)

    tf_records_folder = configuration.tf_records_folder
    output_folder = configuration.output_folder
    make_dirs_safe(output_folder)
    saver_path = output_folder + "/model"

    targets = ["upper_belt", ]
    number_of_targets = len(targets)

    ####################################################################################################################
    # Form computational graph.
    ####################################################################################################################
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            ############################################################################################################
            # Get dataset iterators.
            ############################################################################################################
            dataset_train = data_provider.get_partition(tf_records_folder,
                                                        is_training=True,
                                                        partition_name="train",
                                                        batch_size=configuration.train_batch_size,
                                                        seq_length=configuration.train_seq_length,
                                                        buffer_size=(train_steps_per_epoch + 1))
            dataset_devel = data_provider.get_partition(tf_records_folder,
                                                        is_training=False,
                                                        partition_name="devel",
                                                        batch_size=configuration.devel_batch_size,
                                                        seq_length=configuration.full_seq_length,
                                                        buffer_size=(devel_steps_per_epoch + 1) // 4)
            dataset_test = data_provider.get_partition(tf_records_folder,
                                                       is_training=False,
                                                       partition_name="test",
                                                       batch_size=configuration.test_batch_size,
                                                       seq_length=configuration.full_seq_length,
                                                       buffer_size=(test_steps_per_epoch + 1) // 4)

            iterator_train = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                             dataset_train.output_shapes)
            iterator_devel = tf.data.Iterator.from_structure(dataset_devel.output_types,
                                                             dataset_devel.output_shapes)
            iterator_test = tf.data.Iterator.from_structure(dataset_test.output_types,
                                                            dataset_test.output_shapes)

            next_element_train = iterator_train.get_next()
            next_element_devel = iterator_devel.get_next()
            next_element_test = iterator_test.get_next()

            init_op_train = iterator_train.make_initializer(dataset_train)
            init_op_devel = iterator_devel.make_initializer(dataset_devel)
            init_op_test = iterator_test.make_initializer(dataset_test)

            ############################################################################################################
            # Define placeholders.
            ############################################################################################################
            batch_size_tensor = tf.placeholder(tf.int32)

            audio_train = tf.placeholder(tf.float32, (None, configuration.train_seq_length, 640))
            upper_belt_train = tf.placeholder(tf.float32, (None, configuration.train_seq_length, 1))
            recording_id_train = tf.placeholder(tf.float32, (None, configuration.train_seq_length, 1))

            audio_test = tf.placeholder(tf.float32, (None, configuration.full_seq_length, 640))
            upper_belt_test = tf.placeholder(tf.float32, (None, configuration.full_seq_length, 1))
            recording_id_test = tf.placeholder(tf.float32, (None, configuration.full_seq_length, 1))

            ############################################################################################################
            # Define model graph and get model.
            ############################################################################################################
            with tf.variable_scope("Model"):
                pred_train = core.get_cnn_lstm_model(audio=audio_train,
                                                     batch_size=batch_size_tensor,
                                                     num_layers=configuration.num_layers,
                                                     hidden_units=configuration.hidden_units,
                                                     number_of_outputs=number_of_targets)

            with tf.variable_scope("Model", reuse=True):
                pred_test = core.get_cnn_lstm_model(audio=audio_test,
                                                    batch_size=batch_size_tensor,
                                                    num_layers=configuration.num_layers,
                                                    hidden_units=configuration.hidden_units,
                                                    number_of_outputs=number_of_targets)

            ############################################################################################################
            # Define loss function.
            ############################################################################################################
            tensor_shape_train = [batch_size_tensor, configuration.train_seq_length]
            flattened_size_train = tensor_shape_train[0] * tensor_shape_train[1]

            pred_upper_belt_train = pred_train[:, :, 0]

            single_pred_upper_belt_train = core.flatten_data(pred_upper_belt_train,
                                                             flattened_size_train)

            loss = core.loss_function(pred_upper_belt=single_pred_upper_belt_train,
                                      true_upper_belt=upper_belt_train)

            vars = tf.trainable_variables()
            model_vars = [v for v in vars if v.name.startswith("Model")]
            saver = tf.train.Saver({v.name: v for v in model_vars})

            total_loss = tf.reduce_sum(loss)
            optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate).minimize(total_loss,
                                                                                             var_list=vars)
            ############################################################################################################
            # Initialize variables and perform experiment.
            ############################################################################################################
            sess.run(tf.global_variables_initializer())

            ############################################################################################################
            # Train base model.
            ############################################################################################################
            print("Start training base model.")
            print("Fresh base model.")
            for ee, epoch in enumerate(range(configuration.num_epochs)):
                print("EPOCH:", epoch + 1)
                run_epoch = core.RunEpoch(sess=sess,
                                          partition="train",
                                          init_op=init_op_train,
                                          steps_per_epoch=train_steps_per_epoch,
                                          next_element=next_element_train,
                                          batch_size=configuration.train_batch_size,
                                          seq_length=configuration.train_seq_length,
                                          input_gaussian_noise=configuration.input_gaussian_noise,
                                          optimizer=optimizer,
                                          loss=total_loss,
                                          pred=pred_train,
                                          input_feed_dict={batch_size_tensor: "batch_size",
                                                           audio_train: "audio",
                                                           upper_belt_train: "upper_belt",
                                                           },
                                          targets=targets)

                train_items, train_subject_to_id = run_epoch.run_epoch()

                train_measures = core.get_measures_slope(train_items)
                print("Train pcc:", train_measures["pearson_upper"])

                if ee == 0:
                    best_performance = -1.0

                if (ee) % configuration.val_every_n_epoch == 0:
                    run_epoch = core.RunEpoch(sess=sess,
                                              partition="devel",
                                              init_op=init_op_devel,
                                              steps_per_epoch=devel_steps_per_epoch,
                                              next_element=next_element_devel,
                                              batch_size=configuration.devel_batch_size,
                                              seq_length=configuration.full_seq_length,
                                              input_gaussian_noise=configuration.input_gaussian_noise,
                                              optimizer=None,
                                              loss=None,
                                              pred=pred_test,
                                              input_feed_dict={batch_size_tensor: "batch_size",
                                                               audio_test: "audio",
                                                               upper_belt_test: "upper_belt",
                                                               },
                                              targets=targets)

                    devel_items, devel_subject_to_id = run_epoch.run_epoch()

                    devel_measures = core.get_measures_slope(devel_items)
                    print("Devel pcc:", devel_measures["pearson_upper"])

                    if best_performance < devel_measures["pearson"]:
                        best_performance = devel_measures["pearson"]
                        saver.save(sess, saver_path)

                else:
                    pass

            saver.restore(sess, saver_path)

            run_epoch = core.RunEpoch(sess=sess,
                                      partition="test",
                                      init_op=init_op_test,
                                      steps_per_epoch=test_steps_per_epoch,
                                      next_element=next_element_test,
                                      batch_size=configuration.test_batch_size,
                                      seq_length=configuration.full_seq_length,
                                      input_gaussian_noise=configuration.input_gaussian_noise,
                                      optimizer=None,
                                      loss=None,
                                      pred=pred_test,
                                      input_feed_dict={batch_size_tensor: "batch_size",
                                                       audio_test: "audio"},
                                      targets=targets)

            test_items, test_subject_to_id = run_epoch.run_epoch()

            core.make_label_file(path=output_folder + "/pred.csv",
                                 data=test_items.upper_belt.pred.reshape((16, 6000)))
            print("Best devel pcc:", best_performance)
