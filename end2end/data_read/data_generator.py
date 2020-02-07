import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.io import wavfile

from end2end.configuration import *
from end2end.common import make_dirs_safe

SEQ_LEN = 6000
AUDIO_FRAME_SIZE = 640


def read_data(challenge_folder, partition_to_id):
    belt_data = pd.read_csv(challenge_folder + "/lab/labels.csv")

    data = dict()
    for partition in partition_to_id.keys():
        data[partition] = dict()
        for speaker_id in partition_to_id[partition]:
            data[partition][speaker_id] = dict()

            if speaker_id < 10:
                id_text = "0" + repr(speaker_id)
            else:
                id_text = repr(speaker_id)

            wav = read_recording(challenge_folder + "/wav/" + partition + "_" + id_text + ".wav")

            data[partition][speaker_id]["wav"] = wav
            if partition in ["train", "devel"]:
                data[partition][speaker_id]["upper_belt"] = belt_data[belt_data['filename'] == partition + "_" + id_text + ".wav"]['upper_belt'].values.reshape(-1,).astype(np.float32)

    return data


def read_recording(path):
    fs, data = wavfile.read(path)

    return data.astype(np.float32).reshape((SEQ_LEN, AUDIO_FRAME_SIZE))


def read_belt_signal(path):
    data = np.empty((SEQ_LEN, ), dtype=np.float32)
    with open(path, "r") as fp:
        next(fp)
        counter = 0
        for row in fp:
            clean_row = row.strip().split(",")
            data[counter] = float(clean_row[1])
            counter += 1
    if counter != SEQ_LEN:
        raise ValueError
    return data


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(writer, data, partition, speaker_id):
    if partition in ["train", "devel"]:
        wav = data[partition][speaker_id]["wav"]
        upper_belt = data[partition][speaker_id]["upper_belt"]

        for i, (wav_step,
                upper_belt_step) in enumerate(zip(*(wav, upper_belt))):

            example = tf.train.Example(features=tf.train.Features(feature={
                'sample_id': _int_feature(np.int64(i)),
                'recording_id': _int_feature(np.int64(speaker_id)),
                'upper_belt': _bytes_feature(upper_belt_step.tobytes()),
                'raw_audio': _bytes_feature(wav_step.tobytes()),
            }))

            writer.write(example.SerializeToString())
    elif partition == "test":
        wav = data[partition][speaker_id]["wav"]

        for i, (wav_step) in enumerate(wav):
            example = tf.train.Example(features=tf.train.Features(feature={
                'sample_id': _int_feature(np.int64(i)),
                'recording_id': _int_feature(np.int64(speaker_id)),
                'raw_audio': _bytes_feature(wav_step.tobytes()),
            }))

            writer.write(example.SerializeToString())
    else:
        raise ValueError("Invalid partition selection.")


def main(tf_records_folder, challenge_folder):
    make_dirs_safe(tf_records_folder)
    make_dirs_safe(tf_records_folder + "/train")
    make_dirs_safe(tf_records_folder + "/devel")
    make_dirs_safe(tf_records_folder + "/test")

    partition_to_id = dict()
    partition_to_id["train"] = list(range(17))
    partition_to_id["devel"] = list(range(16))
    partition_to_id["test"] = list(range(16))

    data = read_data(challenge_folder, partition_to_id)

    for partition in partition_to_id.keys():
        print("Making tfrecords for", partition, "partition.")

        for speaker_id in partition_to_id[partition]:
            writer = tf.io.TFRecordWriter(tf_records_folder + "/" + partition + "/" + partition + '{}.tfrecords'.format(speaker_id))
            serialize_sample(writer, data, partition, speaker_id)


if __name__ == "__main__":
    main(TF_RECORDS_FOLDER, CHALLENGE_FOLDER)
