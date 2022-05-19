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
import function as f

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

#####
# 指定したURLから画像取得
def find_image(page_url):
    r = requests.get(page_url)
    soup = BeautifulSoup(r.text, "html.parser")
    print(soup)
    #td = soup.select('image')[0]
    img_tags = soup.find_all("img")
    img_urls = []
    
    for img_tag in img_tags:
      url = img_tag.get("src")
      if url != None:
        img_urls.append(url)
        print(url)


def main():
    # キャプチャを開く
    directory = os.path.dirname(__file__)
    args = f.get_option()
    skip_fps = 1

    if args.type == 'picture' or args.type == 'pic':
        capture = cv2.VideoCapture(os.path.join(directory, args.input)) # 画像ファイル
    elif args.type == 'movie' or args.type == 'mov':
        capture = cv2.VideoCapture(os.path.join(directory, args.input)) # 動画ファイル
        if args.speed == 1:
            read_fps = capture.get(cv2.CAP_PROP_FPS)
            print('frame rate:'+str(read_fps))
            while True:
                if read_fps > 10:
                    read_fps = read_fps / 2
                    skip_fps = skip_fps * 2
                else:
                    break
            
    elif args.type == 'ipcam' or args.type == 'ip':
        capture = cv2.VideoCapture(args.url)
    elif args.type == 'camera' or args.type == 'cam':
        capture = cv2.VideoCapture(0) # カメラ
        if (args.codec == 0):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('B', 'G', 'R', '3'))
        elif (args.codec == 1):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        elif (args.codec == 2):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        elif (args.codec == 3):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FPS, 30)

    elif args.type == 'screenshot' or args.type == 'scr':
        img = ImageGrab.grab()
        img.save('tmp\\tmp.jpg')
        capture = cv2.VideoCapture('tmp\\tmp.jpg') # 画像ファイル

    elif args.type == 'webpage' or args.type == 'web':
        find_image(args.url)
        exit()

    elif args.type == 'scrmov' or args.type == 'scm':
        img = ImageGrab.grab()
        img.save('tmp\\tmp.jpg')
        capture = cv2.VideoCapture('tmp\\tmp.jpg') # 画像ファイル 

    else:
        print("mode is empty")
        exit() 

    if not capture.isOpened():
        print("any images do not exist")
        exit()

    # 特徴を読み込む
    dictionary = []
    face_detector, face_recognizer = f.make_dictionary(dictionary, directory)

    # 時刻情報取得
    ut = time.time()

    # 一連番号
    num = f.get_next_ID(directory)
    frame_counter = 0
    movie_frame_number = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        movie_frame_number += 1
        frame_counter += 1
        if frame_counter < skip_fps:
            continue
        frame_counter = 0

        # 連続スクショ
        if args.type == 'scrmov' or args.type == 'scm':
            img = ImageGrab.grab()
            img.save('tmp\\tmp.jpg')
            capture = cv2.VideoCapture('tmp\\tmp.jpg') # 画像ファイル 

         # フレームをキャプチャして画像を読み込む
        result, image = capture.read()

        if result is False:
            cv2.waitKey(0)
            break

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # 入力サイズを指定する
        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        # 顔を検出する
        result, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            # 顔識別信頼度が一定の値未満なら評価せず
            box = list(map(int, face[:4]))
            confidence = face[-1]
            if confidence < args.accuracy:
                continue

            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            # 辞書とマッチングする
            result, user = f.match(face_recognizer, feature, dictionary, args.resolution)

            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            position = (box[0], box[1] - 10)
            if result:
                # 顔のバウンディングボックスを描画する
                color = (0, 255, 0)
                # 認識の結果を描画する
                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
                id, score = user
                text = "{0} ({1:.2f})".format(id, score)
                if args.type == 'camera' or args.type == 'cam':
                    td = datetime.datetime.now()
                else:
                    td = datetime.timedelta(seconds = movie_frame_number / fps)
                print(str(td) + ' : detect ' + str(user))

            else:
                color = (0, 0, 255)
                # 顔情報の追加更新
                tmp = directory + '\\face_record\\ID_' + f'{num:04}' + '.npy'
                np.save(tmp, feature)
                cv2.imwrite(directory + '\\face_img_record\\ID_' + f'{num:04}.jpg', aligned_face)
                f.add_dictionary(dictionary, tmp)
                print(str(datetime.datetime.now()) + ' : add  ' + 'ID_' + f'{num:04}')

                cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
                id, score = user
                text = 'NEW: ID_' + f'{num:04}'

                num += 1
                if time.time() - ut > 3:
                    PlaySound('sound\\record.wav', SND_ASYNC)
                    ut = time.time()

            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

        # 画像を表示する
        if args.type != 'scrmov' and args.type != 'scm':
            cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
            cv2.imshow("face recognition", image)
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()