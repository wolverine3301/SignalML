# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:58:24 2023

@author: Owner
"""
import librosa #analyze audio signals in general but geared more towards music.
import glob, os
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import math
from tqdm import tqdm
import random
import io

SAMPLE_RATE = 22050 #sample rate
MONO = True #convert stero to mono


'''
loads wave file into chunks returned in an array

params:
    waveFile- path to wav file
    chucksize - in seconds
    pitchAugment - True, creates additional samples by varying the pitch
'''
def genElements(waveFile,chunkSize,pitchAugment = True,saveWavs=False):
    wavChunks = []
    x , sr1 = librosa.load(waveFile,sr=SAMPLE_RATE,mono=MONO)
    tot_sample_duration = librosa.audio.get_duration(x,sr1)
    chunks = math.floor(librosa.audio.get_duration(x,sr1)/chunkSize) #even chunks or clip
    offset = 0
    qbar = tqdm(total=chunks,ncols = 100,colour="#33FFBD", desc ="processing chunks: ",position=0,leave=True)
    for i in range(chunks):
        #qbar.set_description("Processing %s" % waveFile)
        x,sr = librosa.load(waveFile,sr=SAMPLE_RATE,mono=MONO,offset=offset,duration=chunkSize)
        wavChunks.append(x)
        if pitchAugment:
            wavChunks.append(augmentPitchShift(x,sr))
        if saveWavs:
            savewav(waveFile+'_'+str(len(wavChunks)),x )
        offset=offset+chunkSize
        qbar.update(1)
    #qbar.close()
    return wavChunks, chunks, tot_sample_duration
'''
takes folder of raw sounds and creates equal size audio segment samples
returns a dictionary of the samples, the keys are filename_x where x is an iterator on the segments

parameters:
    rawFolder (String) - path to raw clas audio
    sampleDuration (float) - duration of sample in seconds
    
returns:
    sample_dict (dict) - dictionary of samples; keys = sample file names , values = nparray of audio time series
'''
def makeClassSamples(rawFolder,sampleDuration,pitchAugment = True,saveWavs=False):
    
    sample_dict = {}
    offset = 0  #start time to load audio clip
    name_iter = 0
    file_change_flag = ""
    x = 0
    parse_file = True
    filelist = os.listdir(rawFolder)
    cnt = 0
    tot_samples = 0
    tot_duration = 0
    pbar = tqdm(total=len(filelist),ncols = 100,colour="#007824", desc ="Progress: ")
    #iterate through raw audio files
    for file in filelist:
        
        #print(file)
        parse_file = True
        #keep parsing into equal size (time) audio samples until end of file
        name_iter = 0
        offset = 0
        
        pbar.set_description("Processing %s" % file)
        pbar.update(1)
        sample_dict[file],chunks, samp_duration = genElements(rawFolder+'\\'+file, sampleDuration,pitchAugment=pitchAugment,saveWavs=saveWavs)
        tot_duration += samp_duration
        tot_samples = tot_samples + chunks


    pbar.update(1)
    pbar.close()
    
    print("GENERATED "+str(tot_samples)+" SAMPLES")
    print("APPROX "+ str(tot_duration)+" seconds of audio for class")
    #print("APPROX "+ str((sampleDuration*len(sample_dict.values()))/60)+" minutes of audio for class")

    return sample_dict

def makeClassSamples2(rawFolder,sampleDuration):
    
    sample_dict = {}
    offset = 0  #start time to load audio clip
    name_iter = 0
    file_change_flag = ""
    x = 0
    parse_file = True
    filelist = os.listdir(rawFolder)
    cnt = 0

    pbar = tqdm(total=len(filelist),ncols = 100,colour="#007824", desc ="Progress: ")
    #iterate through raw audio files
    for file in filelist:
        
        #print(file)
        parse_file = True
        #keep parsing into equal size (time) audio samples until end of file
        name_iter = 0
        offset = 0
        
        pbar.set_description("Processing %s" % file)
        pbar.update(1)
        while(parse_file):
            #print(file)
            try:
                x , sr = librosa.load(rawFolder+'\\'+file, offset=offset, duration=sampleDuration) #audio time series and sample rate
                #x , sr = librosa.load(os.getcwd()+rawFolder+file)
                #print(x.shape)

            except:
                #print("OH NO")
                parse_file = False
                break
                #continue

            if file != file_change_flag:
                file_change_flag = file
                name_iter = 0
                offset = 0
                sample_dict[file+"_"+str(name_iter)] = x
            else:
                sample_dict[file+"_"+str(name_iter)] = x
                name_iter+=1
                offset+=sampleDuration
                file_change_flag = file
            file_change_flag = file 
            #plt.figure(figsize=(14, 5))
            #librosa.display.waveplot(x, sr=sr)

    pbar.update(1)
    pbar.close()
    print("GENERATED "+str(len(sample_dict.values()))+" SAMPLES")
    print("APPROX "+ str((sampleDuration*len(sample_dict.values())))+" seconds of audio for class")
    print("APPROX "+ str((sampleDuration*len(sample_dict.values()))/60)+" minutes of audio for class")
    return sample_dict

'''
generates a random pitch shift applied to sound data provided
'''
def augmentPitchShift(wav, sr,minShift=-6,maxShift=6):
    x = random.randint(minShift,maxShift)
    y = librosa.effects.pitch_shift(wav, sr=sr, n_steps=x)
    return y

def mixSignals(s1, s2):
    return (s1+s2)/2
'''
classFolder1 = preprocessed class npz file
classFolder2 = preprocessed class npz file
newName = name of new mixed class
percentMix = percent of samples to mix together
'''
def augmentMixedSignals_init(class_dict1,class_dict2,newName,percentMix):

    class1 = []
    class2 = []
    mix_indices = []
    newClass_dict = {}
    newClass_label = newName
    print(newClass_label)
    newClass_dict[newClass_label] = []
    for i in class_dict1.keys():
        for j in class_dict1[i]:
            class1.append(j)
    for x in class_dict2.keys():
        for y in class_dict2[x]:
            class2.append(y)
    if len(class1) > len(class2):
        mix_indices = random.sample(range(1, len(class2)), math.floor(len(class2)*percentMix))
    else:
        mix_indices = random.sample(range(1, len(class1)), math.floor(len(class1)*percentMix))
    #print(mix_indices)
    print("Mixing "+str(len(mix_indices))+ " new samples")
    pbar = tqdm(total=len(mix_indices),ncols = 100,colour="#007824", desc ="Progress: ")
    for mix in mix_indices:
        # MERGE
        newClass_dict[newClass_label].append(mixSignals(class1[mix],class2[mix]))
        pbar.update(1)
    print(len(newClass_dict[newClass_label]))
    pbar.update(1)
    pbar.close()
    return newClass_dict

'''
Creates mixed audio for masking
    Class1 - ground truth or the mask
    Class2 - is class one with mixed in background noise
'''
def generateMaskingElements(class_dict1,class_dict2,newName, saveWavs=False):

    class1 = []
    class2 = []
    mix_indices = []
    newClass_dict = {}
    newClass_label = newName
    print('MASKING:')
    print(newClass_label)
    newClass_dict[newClass_label] = []
    ### load in base classes
    for i in class_dict1.keys():
        for j in class_dict1[i]:
            class1.append(j)

    for x in class_dict2.keys():
        for y in class_dict2[x]:
            class2.append(y)
            
    pbar = tqdm(total=len(mix_indices),ncols = 100,colour="#007824", desc ="Progress: ")
    ##create mix
    for mix in range(len(class1)):
        # MERGE
        index = random.randint(0, len(class2)-1)
        mixedSignal =mixSignals(class1[mix],class2[index])
        newClass_dict[newClass_label].append(mixedSignal)
        if saveWavs:
            savewav('class1_'+str(mix), class1[mix])
            savewav('MIX_'+str(mix), mixedSignal)
        pbar.update(1)
    print(len(class1))
    print(len(newClass_dict[newClass_label]))
    pbar.update(1)
    pbar.close()
    return newClass_dict
'''
saves an numpy file of class data
'''
def saveMELs(mels,filename):
    print('saving ',filename)
    np.savez(filename,*mels.values())  


'''
creates a mel spectagam of the data
'''
def conjourMEL(sample_dict): 
    
    n_fft = 2048
    hop_length = 256
    win_length=n_fft
    n_mels =256
    
    cnt = 0
    MEL = {}
    for example in sample_dict.values():
        for i in example:
            if cnt in MEL.keys():
                MEL[cnt].append(librosa.feature.melspectrogram(i,n_fft=n_fft,hop_length=hop_length,win_length=win_length,n_mels=n_mels))
            else:
                MEL[cnt] = []
                MEL[cnt].append(librosa.feature.melspectrogram(i,n_fft=n_fft,hop_length=hop_length,win_length=win_length,n_mels=n_mels))
                
            cnt = cnt+1
    return MEL

'''
creates a STFT spectagam of the data
'''
def conjourSTFT(sample_dict): 
    
    n_fft = 2048
    hop_length = 256
    win_length=n_fft

    
    cnt = 0
    STFT = {}
    for example in sample_dict.values():
        for i in example:
            if cnt in STFT.keys():
                STFT[cnt].append(librosa.stft(i,n_fft=n_fft,hop_length=hop_length,win_length=win_length))
            else:
                STFT[cnt] = []
                STFT[cnt].append(librosa.stft(i,n_fft=n_fft,hop_length=hop_length,win_length=win_length))
                
            cnt = cnt+1
    return STFT

'''
classFolder1 = preprocessed class npz file
classFolder2 = preprocessed class npz file
newName = name of new mixed class
percentMix = percent of samples to mix together
'''
def augmentMixedSignals(classFolder1,classFolder2,newName,percentMix):

    #load data
    data1 = np.load(classFolder1)
    data2 = np.load(classFolder2)
    
    class1 = []
    class2 = []
    mix_indices = []
    newClass_dict = {}
    newClass_label = newName
    print(newClass_label)
    newClass_dict[newClass_label] = []
    for i in data1.keys():
        for j in data1.get(i):
            class1.append(j)
    for x in data2.keys():
        for y in data2.get(x):
            class2.append(y)
    if len(class1) > len(class2):
        mix_indices = random.sample(range(1, len(class2)), math.floor(len(class2)*percentMix))
    else:
        mix_indices = random.sample(range(1, len(class1)), math.floor(len(class1)*percentMix))
    #print(mix_indices)
    print("Mixing "+str(len(mix_indices))+ " new samples")
    print()
    pbar = tqdm(total=len(mix_indices),ncols = 100,colour="#007824", desc ="Progress: ")
    for mix in mix_indices:
        t1 = librosa.feature.inverse.mel_to_audio(class1[mix])
        t2 = librosa.feature.inverse.mel_to_audio(class2[mix])
        # MERGE
        merge = (t1+t2)/2
        #merge = librosa.effects.time_stretch(merge, rate=2.0)
        newClass_dict[newClass_label].append(merge)
        pbar.update(1)
    print(len(newClass_dict[newClass_label]))
    pbar.update(1)
    pbar.close()
    mels = conjourMEL(newClass_dict)
    saveMELs(mels,newClass_label)
    
'''modify amplitude (volume) of a wav'''
def modify_amplitude(wav,percent):
    for s in range(0,len(wav)):
        wav[s] = wav[s] * percent
        
def savewav(name,x):
    sr = 22050 # sample rate
    sf.write(name+'.wav', x,sr)
    
    

'''
def conjourChromagram(sample_dict):
    chromagram = librosa.feature.chroma_stft(x, sr=sr)
def traceHarmonics(sample_dict):
    harmonics = librosa.effects.harmonic(x)

def traceZeroCrossing(sample_dict):
    zero_crossings = librosa.zero_crossings(x, pad=False)

def traceSpectralRolloff(sample_dict):
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)
def traceSpectralCentroids(sample_dict):
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)    
    
mfcc = librosa.feature.melspectrogram(x)


mfcc = np.rot90(mfcc)
chromagram = np.rot90(chromagram)
spectral_rolloff = np.rot90(spectral_rolloff)
spectral_centroids = np.rot90(spectral_centroids)

'''

'''
a = makeClassSamples(os.getcwd()+'\\raw\\birds', 0.5)
b = makeClassSamples(os.getcwd()+'\\raw\\instramentalMusic', 0.5)
c = makeClassSamples(os.getcwd()+'\\raw\\singing', 0.5)
d = makeClassSamples(os.getcwd()+'\\raw\\vehicle', 0.5)
e = makeClassSamples(os.getcwd()+'\\raw\\voice', 0.5)

ea = augmentMixedSignals_init(e,d,"voiceVehicle",0.1)
eb = augmentMixedSignals_init(e,a,"voicebirds",0.1)
ec = augmentMixedSignals_init(e,d,"voiceinstramentalMusic",0.1)

ca = augmentMixedSignals_init(c,d,"singingVehicle",0.1)
cb = augmentMixedSignals_init(c,d,"singinginstramentalMusic",0.1)

MEL = conjourMEL(a)
saveMELs(MEL,'birds')  
MEL = conjourMEL(b)
saveMELs(MEL,'instramentalMusic')
MEL = conjourMEL(c)
saveMELs(MEL,'singing')  
MEL = conjourMEL(d)
saveMELs(MEL,'vehicle')  
MEL = conjourMEL(e)
saveMELs(MEL,'voice')
MEL = conjourMEL(ea)
saveMELs(MEL,'voiceVehicle')  
MEL = conjourMEL(eb)
saveMELs(MEL,'voiceBird')  
MEL = conjourMEL(ec)
saveMELs(MEL,'voiceMusic')  
MEL = conjourMEL(ca)
saveMELs(MEL,'singingVehicle')  
MEL = conjourMEL(cb)
saveMELs(MEL,'singingmusic')  
'''

#creating masking dataset
'''
b = makeClassSamples(os.getcwd()+'\\raw\\instramentalMusic', 1)
c = makeClassSamples(os.getcwd()+'\\raw\\singing',1)
d = makeClassSamples(os.getcwd()+'\\raw\\vehicle', 1)
e = makeClassSamples(os.getcwd()+'\\raw\\voice', 1)
MEL = conjourMEL(e)
saveMELs(MEL,'voice')
MEL = conjourMEL(c)
saveMELs(MEL,'singing')  
ab = generateMaskingElements(e,d,'voiceVehicle')
MEL = conjourMEL(ab)
saveMELs(MEL,'voiceVehicle')  
ac = generateMaskingElements(e,b,'voiceMusic')
MEL = conjourMEL(ac)
saveMELs(MEL,'voiceMusic')  
ad = generateMaskingElements(c,d,'singVehicle')
MEL = conjourMEL(ad)
saveMELs(MEL,'singVehicle')  
ae = generateMaskingElements(c,b,'singMusic')
MEL = conjourMEL(ae)
saveMELs(MEL,'singMusic')  
'''
##############################################
'''
b = makeClassSamples(os.getcwd()+'\\raw\\instramentalMusic', 1)
c = makeClassSamples(os.getcwd()+'\\raw\\singing',1)
d = makeClassSamples(os.getcwd()+'\\raw\\vehicle', 1)
e = makeClassSamples(os.getcwd()+'\\raw\\voice', 1)
MEL = conjourSTFT(e)
saveMELs(MEL,'voice')
MEL = conjourSTFT(c)
saveMELs(MEL,'singing')  
ab = generateMaskingElements(e,d,'voiceVehicle')
MEL = conjourSTFT(ab)
saveMELs(MEL,'voiceVehicle')  
ac = generateMaskingElements(e,b,'voiceMusic')
MEL = conjourSTFT(ac)
saveMELs(MEL,'voiceMusic')  
ad = generateMaskingElements(c,d,'singVehicle')
MEL = conjourSTFT(ad)
saveMELs(MEL,'singVehicle')  
ae = generateMaskingElements(c,b,'singMusic')
MEL = conjourSTFT(ae)
saveMELs(MEL,'singMusic') 
'''
#####################################################
''' TESTING '''
'''
a = makeClassSamples(os.getcwd()+'\\rawTest\\birds', 1)
b = makeClassSamples(os.getcwd()+'\\rawTest\\instramentalMusic',1)
c = makeClassSamples(os.getcwd()+'\\rawTest\\singing', 1)
d = makeClassSamples(os.getcwd()+'\\rawTest\\vehicle', 1)
e = makeClassSamples(os.getcwd()+'\\rawTest\\voice', 1)


MEL = conjourMEL(e)
saveMELs(MEL,'voice')
MEL = conjourMEL(c)
saveMELs(MEL,'singing')  


ab = generateMaskingElements(e,d,'voiceVehicle',saveWavs=True)
MEL = conjourMEL(ab)
saveMELs(MEL,'voiceVehicle')  
ac = generateMaskingElements(e,b,'voiceMusic',saveWavs=True)
MEL = conjourMEL(ac)
saveMELs(MEL,'voiceMusic')  
ad = generateMaskingElements(c,d,'singVehicle')
MEL = conjourMEL(ad)
saveMELs(MEL,'singVehicle')  
ae = generateMaskingElements(c,b,'singMusic')
MEL = conjourMEL(ae)
saveMELs(MEL,'singMusic') 
'''
a = makeClassSamples(os.getcwd()+'\\raw\\birds', 1)
b = makeClassSamples(os.getcwd()+'\\raw\\instramentalMusic',1)
c = makeClassSamples(os.getcwd()+'\\raw\\singing', 1)
d = makeClassSamples(os.getcwd()+'\\raw\\vehicle', 1)
e = makeClassSamples(os.getcwd()+'\\raw\\voice', 1)


MEL = conjourMEL(e)
saveMELs(MEL,'voice')
MEL = conjourMEL(c)
saveMELs(MEL,'singing')  


ab = generateMaskingElements(e,d,'voiceVehicle')
MEL = conjourMEL(ab)
saveMELs(MEL,'voiceVehicle')  
ac = generateMaskingElements(e,b,'voiceMusic')
MEL = conjourMEL(ac)
saveMELs(MEL,'voiceMusic')  
ad = generateMaskingElements(c,d,'singVehicle')
MEL = conjourMEL(ad)
saveMELs(MEL,'singVehicle')  
ae = generateMaskingElements(c,b,'singMusic')
MEL = conjourMEL(ae)
saveMELs(MEL,'singMusic') 
#saveMELs(MEL,'instramentalMusic')  

#augmentMixedSignals(os.getcwd()+'\\extract\\instramentalMusic.npz',os.getcwd()+'\\extract\\voice.npz','voiceMusic',0.05)
#augmentMixedSignals(os.getcwd()+'\\extract\\vehicle.npz',os.getcwd()+'\\extract\\singing.npz','singingVehicle',0.03)

#samples = loadFiles()



def makeSTFT(samples):
    for key in samples.keys():
        samples[key] = librosa.stft(samples[key],n_fft=1024,hop_length=256,win_length=1024)
    

#x , sr = librosa.load(os.getcwd()+'\\raw\\birds\\Ayamhutanrembau.mp3', duration=5) #audio time series and sample rate
def makewav(name):
    sr = 22050 # sample rate
    T = 5.0    # seconds
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz
    #Playing the audio
    #ipd.Audio(x, rate=sr) # load a NumPy array
    #Saving the audio
    librosa.output.write_wav('tone_220.wav', x, sr)
#makeClassSamples("\\raw\\birds\\",5)
    