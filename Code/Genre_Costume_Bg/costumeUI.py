import gradio as gr
import cohere
import io
emotion ="neutral"
from PIL import Image
from stability_sdk import client
import pickle
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import operator
from collections import defaultdict


def distance(instance1, instance2, k):  #mahalanobis distance
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance,trainingset[x],k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#function to identify the nearest neighbors
def nearestclass(neighbors):
    classVote = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
            
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def get_genre():
    dataset1=[]
    with open('Data/mydataset1.dat','rb') as f:
            while True:
                try:
                    dataset1.append(pickle.load(f))
                except EOFError:
                    f.close()
                    break
                

    results = defaultdict(int)

    directory = "Data/genres_original"

    i = 1
    for folder in os.listdir(directory):
        results[i] = folder
        # print(results[i])
        i += 1
    (rate,sig)=wav.read("Data/song.wav")
    mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    featurenew=(mean_matrix,covariance,0)
    pred=nearestclass(getNeighbors(dataset1 ,featurenew , 5))
    return results[pred]

def define_tokens():
    #co=cohere.Client("0hAYMWX3qLVQ8aWTGekrAP3IijxdtrzXatLJoG4K")
    co=cohere.Client("6In9zdzgZGWvKCavZDYt1gOsji12jKnzmWlwe7il")
    # stability=client.StabilityInference(key="sk-GJaICmqVMR54p3o5s72uliymou4QiPk7lzLQTJpzOEdietSx")
    stability=client.StabilityInference(key="sk-VZb13R5HmrD7dW1NkOraAxaUR7YvvDTLaXHPw35dmDEgHKtI")
    
    return co,stability

def prompt_generator_bg(co,stability,genre):
    #prompt="describe an empty background without any people for a"+ genre+" dance routine"
    #prompt="i need brief phrases describing of an appropriate background based on the genre of the song. can you mention the lighting and place accordingly for a dance performance .give it in the form of continuous keywords. there should be no description of people, only the background. the genre is " +genre
    prompt= "describe a single particular empty background setting for a "+genre+" dance routine. give it in the form of continuous keywords, not sentences. there should be no people"
    predicted=co.generate(prompt=prompt,max_tokens=80)
    #print(predicted[0])
    img=generate_img(predicted[0],stability)
    return img


def prompt_generator_costume(co,stability,genre,emotion):
    prompt="i will give you a song genre and a particular emotion related to the song. you need to suggest clothes suitable to dance for this song. mention the type of clothes (tops, bottoms, skirts, shorts, dresses), the material for them (satin, velvet, jeans, cotton), and the colors and patterns on the clothes. i want to search for clothes in an online clothing store. instead of sentences, can you give me keywords that i can enter into the search tab of the clothing website? genre:" + genre + " emotion: "+ emotion
    predicted=co.generate(prompt=prompt) #(prompt=prompt,max_tokens=80)
    img=generate_img(predicted[0],stability)
    return img

def generate_img(prompt,stability):
    img_generations=stability.generate(prompt)
    for img in img_generations:
        for artifact in img.artifacts:
            if artifact.type==1:
                img=Image.open(io.BytesIO(artifact.binary))
                #img.show()
                return img
            
def prompt(genre,emotion):
    co,stability=define_tokens()
    genre=get_genre()
    res1=prompt_generator_costume(co,stability,"metal","angry")
    res2=prompt_generator_bg(co,stability,"metal")
    return res1,res2


if __name__=="__main__":  #takes around 2mins 40secs to give result
    demo = gr.Interface(fn=prompt,inputs=["audio","video"],outputs=["image","image"],title="Costume & Background Generator",description="Upload your song in wav format!")
    demo.launch(share=True)   
    #co,stability=define_tokens()
    #prompt_generator_bg(co,stability,"hiphop")
  
