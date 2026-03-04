# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:58:49 2023

@author: Logan Collier
"""

from os import walk
import os
import sys
#from __future__ import unicode_literals
#import youtube_dl
import ffmpeg
import subprocess
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import numpy as np
import time


'''
getsong - inputs: 
    Link - a url link to the song
    path - path to save the audio
    
    returns list of video ids
'''
#path = "C:\\Users\\Owner\\Desktop\\phonem\\musicRaw"
def getSong(Link, path):
    vidIDs =[]
    try:
        link = Link
    except IndexError:
            scriptName = sys.argv[0]
            print("Usage: python " + scriptName + " linkOfVideo")
            exit()
    #Change this path with yours.
    #Also make sure that youtube-dl and ffmpeg installed.
    #Previous versions of youtube-dl can be slow for downloading audio. Make sure you have downloaded the latest version from webpage.
    #https://github.com/rg3/youtube-dl
    print(path)
    #os.chdir(path)
    #os.system("youtube-dl --extract-audio " + link)
    print(link)
    #os.system("yt-dlp -x " + link)
    os.system("yt-dlp --extract-audio --audio-format mp3 --audio-quality 0 " +link)
    #print(link.strip())
    vidID= link
    
    vidIDs.append(vidID)
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break
    for i in range(0, len(f)):
            if ".opus" in f[i] and vidID in f[i]:
                vidName = f[i].strip()
                #print(vidName)
                cmdstr = "ffmpeg -i raw/"+vidName+" -ab 160k -ac 2 -ar 44100 -vn "+vidID+".wav"
                #cmdstr = "ffmpeg -i \"" + vidName + "\" -f wav -flags bitexact \"" + vidName[:-5] + ".wav"  + "\""
                #print(cmdstr)
                subprocess.call(cmdstr, shell=True)
                #os.system(cmdstr)
                #os.remove(vidName) #Will remove original opus file. Comment it if you want to keep that file.
    return vidIDs
'''
converts file format to a wav file
reads in a file of video URLs to pull from youtube

'''
def toWav(extractFile,savepath):
    if not os.path.exists(os.getcwd()+savepath):
        print("NOT EXSIST: "+ os.getcwd()+savepath)
        os.makedirs(os.getcwd()+savepath)
    f = open('raw/'+extractFile)
    urls = f.readlines()

    #print(urls)
    cnt = 0
    for line in urls:

        getSong(line,os.getcwd()+savepath)
        
        time.sleep(1)
        cnt = cnt+1
    f.close()
    
'''
uses spleeter to seperate wave files into stems
'''
def spleet(songfile,vidId,extractFloder):
    #cmd = 'spleeter separate -i '+ songfile+' -p spleeter:2stems -o test'
    cmd = 'python -m spleeter separate -i '+songfile+' -p spleeter:2stems -o '+extractFloder
    os.system(cmd)

#getSong("https://www.youtube.com/watch?v=dQw4w9WgXcQ", os.getcwd()+"\\raw")
#os.system("yt-dlp -x https://www.youtube.com/watch?v=dQw4w9WgXcQ")
#os.system("yt-dlp -f 'ba' -x --audio-format mp3 https://www.youtube.com/watch?v=dQw4w9WgXcQ  -o YOYOYO.%(ext)s")
toWav("birdsURL.txt","\\raw\\birds")
def run(URLFILE):
    cnt=0
    f = os.getcwd()+"\\raw"
    extractPath = f + "\\extract"
    wave = ''
    mypath = f
    if not os.path.exists(f):
        os.makedirs(f)
        os.makedirs(extractPath)
    
    os.chdir(mypath)
    

    for file in os.listdir(f):
        if file.endswith(".wav"):
            name =file[:-4]
            if name in os.listdir(extractPath):
                print('already processed')
                continue
            else:
                wave = os.path.join(f,file)
                newS = list(wave)
                newS.insert(0, '"')
                newS.append('"')
                wave = ''.join(newS)
                print(wave)
                spleet(wave,str(cnt))       
#run()