from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import tensorflow as tf

AUDIO_FRAME_SIZE = 640


def get_partition(dataset_dir,
                  is_training,
                  partition_name,
                  batch_size,
                  seq_length,
                  buffer_size):
    root_path = Path(dataset_dir) / partition_name
    paths = [str(x) for x in root_path.glob('*.tfrecords')]

    dataset = tf.data.TFRecordDataset(paths)

    if partition_name in ["train", "devel"]:
        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features={
                                                                    'sample_id': tf.FixedLenFeature([], tf.int64),
                                                                    'recording_id': tf.FixedLenFeature([], tf.int64),
                                                                    'raw_audio': tf.FixedLenFeature([], tf.string),
                                                                    'upper_belt': tf.FixedLenFeature([], tf.string),
                                                                }
                                                                ))

        dataset = dataset.map(lambda x: {
            'sample_id': tf.cast(tf.reshape(x['sample_id'], (1,)), tf.int32),
            'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
            'raw_audio': tf.reshape(tf.decode_raw(x['raw_audio'], tf.float32), (AUDIO_FRAME_SIZE,)),
            'upper_belt': tf.reshape(tf.decode_raw(x['upper_belt'], tf.float32), (1,)),
        })
    elif partition_name == "test":
        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features={
                                                                    'sample_id': tf.FixedLenFeature([], tf.int64),
                                                                    'recording_id': tf.FixedLenFeature([], tf.int64),
                                                                    'raw_audio': tf.FixedLenFeature([], tf.string),
                                                                }
                                                                ))

        dataset = dataset.map(lambda x: {
            'sample_id': tf.cast(tf.reshape(x['sample_id'], (1,)), tf.int32),
            'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
            'raw_audio': tf.reshape(tf.decode_raw(x['raw_audio'], tf.float32), (AUDIO_FRAME_SIZE,)),
        })
    else:
        raise ValueError("Invalid partition selection.")

    dataset = dataset.repeat()
    dataset = dataset.batch(seq_length)
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset
