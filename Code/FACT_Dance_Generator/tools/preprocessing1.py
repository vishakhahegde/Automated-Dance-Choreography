from absl import app
from absl import flags
from absl import logging

import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import librosa
from loader import AISTDataset

import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '/content/drive/MyDrive/Capstone_Project/aistplusplus_api-main/aist_plusplus_final',  #SEED MOTION
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'audio_dir', '/content/drive/MyDrive/Capstone_Project/mint-main/tools',  #STORE INPUT AUDIO FILE IN TOOLS
    'Path to the input audio wav file.')
flags.DEFINE_string(
    'audio_cache_dir', '/content/drive/MyDrive/Capstone_Project/mint-main/tools/audio_feats/',  #STORE AUDIO FEATURES 
    'Path to cache dictionary for audio features of input audio.')
flags.DEFINE_string(
    'tfrecord_path', '/content/drive/MyDrive/Capstone_Project/mint-main/tools',  #OUTPUT PATH SINGLE BINARY FILE
    'Output path for the tfrecord files of input audio (binary).')

# RNG = np.random.RandomState(42)


def create_tfrecord_writer(output_file):
    writer = []
    writer.append(tf.io.TFRecordWriter("{}/weeknd".format(output_file)))
    # for i in range(n_shards):
    #     writers.append(tf.io.TFRecordWriter(
    #         "{}-{:0>5d}-of-{:0>5d}".format(output_file, i, n_shards)
    #     ))
    return writer


def close_tfrecord_writer(writer):
        writer[0].close()


def write_tfexample(writer, tf_example):
    print("writing to file")
    writer[0].write(tf_example.SerializeToString())
    print(tf_example.SerializeToString())


def to_tfexample(motion_sequence, audio_sequence, motion_name, audio_name):
    print("concatenating audio and motion features")
    features = dict()
    features['motion_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[motion_name.encode('utf-8')]))
    features['motion_sequence'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=motion_sequence.flatten()))
        
    #print("MOTION SEQUENCE",features['motion_sequence'])
    
    features['motion_sequence_shape'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=motion_sequence.shape))
    features['audio_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[audio_name.encode('utf-8')]))
    features['audio_sequence'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=audio_sequence.flatten()))
    features['audio_sequence_shape'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=audio_sequence.shape))
    #print("FEATURESSSS", features)
    example = tf.train.Example(features=tf.train.Features(feature=features))
    print(example)
    return example


def load_cached_audio_features(audio_name):
    #audio_name=
    print("loading audio features")
    return np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")), audio_name


def cache_audio_features(audio_name):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        y, sr = librosa.load(os.path.join(FLAGS.audio_dir, "{}.wav".format(audio_name))) 
        # Compute the tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo

    save_path = os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")
    # if os.path.exists(save_path):
    #     continue
    data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{audio_name}.wav"), sr=SR)
    #print("Data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    #print("Envelope shape", envelope[:,None].shape)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    #print("mfcc shape", mfcc[:,None].shape)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)
    #print("chroma shape", chroma.shape)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
        start_bpm=_get_tempo(audio_name), tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate([
        envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
    ], axis=-1)
    np.save(save_path, audio_feature)


def main(_):
    os.makedirs(os.path.dirname(FLAGS.tfrecord_path), exist_ok=True)
    tfrecord_writers = create_tfrecord_writer(FLAGS.tfrecord_path)

    # create list of seed motion names present in crossmodal_test
    seq_names = [] 
    seq_names += np.loadtxt(
        os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
    ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]
    num=random.randint(1,20) #choose a random seed motion from seq_names

    # create audio features
    print ("Pre-compute audio features ...")
    os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
    #cache_audio_features(seq_names[num])
    cache_audio_features("weeknd")
    
    # load data
    dataset = AISTDataset(FLAGS.anno_dir)
    #print("dataset",dataset[0])
    n_samples = 1
    #for i, seq_name in enumerate(seq_names):
    #logging.info("processing %d / %d" % (i + 1, n_samples))

    '''smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
        dataset.motion_dir, seq_names[num]) #CHANGE MOTION_DIR TO DIR WITH ONLY HIPHOP SEED MOTIONS
    smpl_trans /= smpl_scaling
    smpl_poses = R.from_rotvec(
        smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)
    smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
    #print("MOTION SEQUENCE SHAPE", smpl_motion.shape)
    audio, audio_name = load_cached_audio_features(seq_name)

    tfexample = to_tfexample(smpl_motion, audio, seq_name, audio_name)
    write_tfexample(tfrecord_writers, tfexample)'''

    # If testval, also test on un-paired data
    
    logging.info("Also add un-paired motion-music data for testing.")
    #for i, seq_name in enumerate(seq_names * 10):
    #logging.info("processing %d / %d" % (i + 1, n_samples * 10))
    print("Sequence names of nums", seq_names[num])
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(dataset.motion_dir, seq_names[num])
    smpl_trans /= smpl_scaling
    smpl_poses = R.from_rotvec(
        smpl_poses.reshape(-1, 3)).as_matrix().reshape(smpl_poses.shape[0], -1)
    smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
    print("smpl motion shape", smpl_motion.shape)
    
    audio, audio_name = load_cached_audio_features("weeknd") 
    print("did it work")
    tfexample = to_tfexample(smpl_motion, audio, seq_names[num], audio_name)
    write_tfexample(tfrecord_writers, tfexample)
    print("did it write")
    close_tfrecord_writer(tfrecord_writers)
    
if __name__ == '__main__':
  app.run(main)