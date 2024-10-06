# -*- coding: utf-8 -*-
"""moodclassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HN2Iv7JcRjen3J6X1eQVXciuuwlu6YMA
"""

#Libraries to pre-process the variables
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/MER/data_moods.csv")

#Define the features and the target
col_features = df.columns[6:-3]
X = df[col_features]
Y = df['mood']
#Normalize the features
X= MinMaxScaler().fit_transform(X)
#Encode the labels (targets)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)
#Split train and test data with a test size of 20%
X_train,X_test,Y_train,Y_test = train_test_split(X,encoded_y,test_size=0.2,random_state=15)

#Libraries to create the Multi-class Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#Function that creates the structure of the Neural Network
def base_model():
    #Create the model
    model = Sequential()
#Add 1 layer with 8 nodes,input of 4 dim with relu function
    model.add(Dense(8,input_dim=10,activation='relu'))
#Add 1 layer with output 3 and softmax function
    model.add(Dense(4,activation='softmax'))
#Compile the model using logistic loss function and adam     optimizer, accuracy correspond to the metric displayed
    model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
    return model
#Configure the estimator with 300 epochs and 200 batchs. the build_fn takes the function defined above.
estimator = KerasClassifier(build_fn=base_model,epochs=300,
                            batch_size=200)

#Library to evaluate the model
from sklearn.model_selection import cross_val_score, KFold

#Evaluate the model using KFold cross validation
kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(estimator,X,encoded_y,cv=kfold)
print("%.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

#Train the model with the train data
estimator.fit(X_train,Y_train)
#Predict the model with the test data
y_preds = estimator.predict(X_test)

X_test

print(type(X_test[0]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Libraries to create the multiclass model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#Import tensorflow and disable the v2 behavior and eager mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

#Library to validate the model
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import librosa
l=[]
#audios=["/content/drive/MyDrive/MER/mWA5.wav","/content/drive/MyDrive/MER/mBR2.wav", "/content/drive/MyDrive/MER/mHO5.wav", "/content/drive/MyDrive/MER/mJB0.wav", "/content/drive/MyDrive/MER/mJS3.wav", "/content/drive/MyDrive/MER/mKR1.wav","/content/drive/MyDrive/MER/mLO2.wav", "/content/drive/MyDrive/MER/mPO2.wav"]
audios=["/content/drive/MyDrive/Capstone_Project/aistplusplus_api-main/aist_plusplus_final/sub_songs/subsong_0.wav", "/content/drive/MyDrive/Capstone_Project/aistplusplus_api-main/aist_plusplus_final/sub_songs/subsong_1.wav", "/content/drive/MyDrive/Capstone_Project/aistplusplus_api-main/aist_plusplus_final/sub_songs/subsong_2.wav", "/content/drive/MyDrive/Capstone_Project/aistplusplus_api-main/aist_plusplus_final/sub_songs/subsong_3.wav"]
# Load audio file
for i in audios:
  audio_file = i
  audio, sr = librosa.load(audio_file)

  # Calculate length in seconds
  length = librosa.get_duration(y=audio, sr=sr)

  # Calculate danceability
  tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
  beat_times = librosa.frames_to_time(beat_frames, sr=sr)
  diff_beat_times = beat_times[1:] - beat_times[:-1]
  danceability = sum(diff_beat_times) / len(diff_beat_times)

  # Calculate acousticness
  spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
  rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
  acousticness = sum(spectral_centroid) / sum(rolloff)

  # Calculate energy
  energy = sum(audio ** 2)

  # Calculate instrumentalness
  harmonic = librosa.effects.harmonic(audio)
  percussive = librosa.effects.percussive(audio)
  RMS_harmonic = librosa.feature.rms(y=harmonic)[0]
  RMS_percussive = librosa.feature.rms(y=percussive)[0]
  instrumentalness = sum(RMS_percussive) / (sum(RMS_harmonic) + sum(RMS_percussive))

  # Calculate liveness
  onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
  D = librosa.stft(audio)
  chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
  hop_length = 512
  beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
  Csync = librosa.util.sync(D, beat_frames, aggregate=np.median)
  beat_onset_env = librosa.util.sync(onset_env, beat_frames, aggregate=np.mean)
  liveness = sum(beat_onset_env) / len(beat_onset_env)

  # Calculate valence
  valence = librosa.feature.rms(y=audio)

  # Calculate loudness
  loudness = librosa.amplitude_to_db(librosa.feature.rms(y=audio))

  # Calculate speechiness
  mfcc = librosa.feature.mfcc(y=audio, sr=sr)
  speechiness = sum(mfcc[1:6]) / sum(mfcc)

  # Calculate tempo
  tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
  l.append([length,danceability,acousticness,energy,instrumentalness,liveness,np.mean(valence),np.mean(loudness),np.mean(speechiness),tempo])

import numpy

j=[length,danceability,acousticness,energy,instrumentalness,liveness,np.mean(valence),np.mean(loudness),np.mean(speechiness),tempo]

k=[length,danceability,acousticness,energy,instrumentalness,liveness,np.mean(valence),np.mean(loudness),np.mean(speechiness),tempo]

for i in j:
  print(i)

output=l
df1 = pd.DataFrame(output)
df1

scaled =  MinMaxScaler()
new_scaled=scaled.fit_transform(np.array(output))
new_scaled

new=np.array(new_scaled)
new

y_preds = estimator.predict(new)

y_preds

emolabels=["calm","energetic","happy","sad"]
emo=[0,1,2,3]
l=[]
f = open("/content/drive/MyDrive/Capstone_Project/mint-main/tools/emo_codes.txt", "r")
file=f.readlines()
for i in file:
  no=int(i.strip())
  l.append(no)
print(l)