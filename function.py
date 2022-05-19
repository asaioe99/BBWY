# coding: UTF-8
import os
import sys
import glob
import pafy
from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH
import numpy as np
import time
import datetime
import cv2
from winsound import PlaySound, SND_PURGE, SND_ASYNC
from argparse import ArgumentParser
from PIL import ImageGrab
#####
import requests
from bs4 import BeautifulSoup
#####

# コマンドライン引数のパーサー
def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('type', type=str, default='picture', choices=['scrmov', 'scm','picture', 'pic', 'movie', 'mov', 'ipcam', 'ip', 'camera', 'cam', 'screenshot', 'scr', 'webpage', 'web'], help = 'set the input type')
    argparser.add_argument('-u', '--url', type=str, default='', help = 'an URL of IPCam')
    argparser.add_argument('-i', '--input', type=str, default='', help = 'a path of input file like: jpg, wav, png')
    argparser.add_argument('-d', '--device', type=int, default=0, help = 'device number')
    argparser.add_argument('-s', '--speed', type=int, default=0, help = '1: frame skip mode')
    argparser.add_argument('-r', '--resolution', type=float, default=0.363, help = 'resolution of faces')
    argparser.add_argument('-a', '--accuracy', type=float, default=0.995, help = 'face detection accuracy: 0.0 ~ 1.0')
    argparser.add_argument('-c', '--codec', type=int, default=2, help = 'camera codec format: 0 = BGR3 , 1 = MJPG, 2 = H264, 3 = YUYV') 
    args = argparser.parse_args()
    return args

    # 登録したデータの中から次の一連番号を取得
def get_next_ID(directory):
    max = 0
    directory += '\\face_record\\'
    files = glob.glob(os.path.join(directory, "*.npy"))
    if files is None:
        return 0
    for file in files:
        user_id = os.path.splitext(os.path.basename(file))[0]
        t = int(user_id.split('_')[1])
        if t > max:
            max = t
    return max + 1

    # 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
def match(recognizer, feature1, dictionary, r):
    max_score = -1.0
    max_user_id = ''
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= max_score:
            max_score = score
            max_user_id = user_id

    if max_score > r:
        return True, (max_user_id, max_score)
    else:
        return False, (max_user_id, max_score)

    # 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
def match_all(recognizer, feature1, dictionary, r):
    match_list = []
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= r:
            #print((user_id, score))
            match_list.append((user_id, score))

    if match_list:
        return True, match_list
    else:
        return False, match_list

def make_dictionary(dictionary, directory):
    directory1 = directory + '\\face_record\\'
    directory2 = directory + '\\model\\'

    files = glob.glob(os.path.join(directory1, "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))
        print('loading face data:' + user_id)

    # モデルを読み込む
    weights = os.path.join(directory2, "yunet.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))
   # weights = os.path.join(directory2, "arcfaceresnet100-8.onnx")
    weights = os.path.join(directory2, "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    return face_detector, face_recognizer

def add_dictionary(dictionary, newID):
    feature = np.load(newID)
    user_id = os.path.split(os.path.basename(newID))[1]
    dictionary.append((user_id, feature))