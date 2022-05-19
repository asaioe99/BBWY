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
import function as f

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128
    
def main():
    args = f.get_option()
    skip_fps = 1
    # キャプチャを開く
    directory = os.path.dirname(__file__)

    if args.type == 'picture' or args.type == 'pic':
        capture = cv2.VideoCapture(os.path.join(directory, args.input)) # 画像ファイル
    elif args.type == 'movie' or args.type == 'mov':
        capture = cv2.VideoCapture(os.path.join(directory, args.input)) # 動画ファイル
        if args.speed == 1:
            read_fps = capture.get(cv2.CAP_PROP_FPS)
            print('frame rate:' + str(read_fps))
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
    else:
        print("set is empty")
        exit()         

    if not capture.isOpened():
        print("cv2.VideoCapture error")
        exit()

    # 特徴を読み込む
    dictionary = []
    files = glob.glob(os.path.join(directory + '\\face_find\\', "*.npy"))
    for file in files:
        feature = np.load(file)
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))
        print('loading face data:' + user_id)

    # モデルを読み込む
    face_detector, face_recognizer = f.make_dictionary(dictionary, directory)

    ut = time.time()
    frame_counter = 0
    movie_frame_number = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # フレームをキャプチャして画像を読み込む
        result, image = capture.read()
        frame_counter += 1
        if frame_counter < skip_fps:
            continue
        frame_counter = 0
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
            # 顔を切り抜き特徴を抽出する
            aligned_face = face_recognizer.alignCrop(image, face)
            feature = face_recognizer.feature(aligned_face)

            # 辞書とマッチングする
            result, user = f.match(face_recognizer, feature, dictionary, args.resolution)
            
            result_all, persons = f.match_all(face_recognizer, feature, dictionary, args.resolution)
            print('detect ' + str(user))
            if result_all:
                persons.sort(reverse=True, key = lambda x: x[1])
                for person in persons:
                    print(person)

            # 顔のバウンディングボックスを描画する
            box = list(map(int, face[:4]))
            # 顔識別信頼度が一定の値未満なら評価せず
            confidence = face[-1]
            if confidence < args.accuracy:
                continue
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            # 認識の結果を描画する
            id, score = user
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

            if result:
                if (args.type == 'movie' or args.type == 'mov'):# and float(time.time - ut) > 3:
                    PlaySound('sound\\detect.wav', SND_ASYNC)
                if (args.type == 'camera' or args.type == 'cam'):
                    td = datetime.datetime.now()
                else:
                    td = datetime.timedelta(seconds = movie_frame_number / fps)
                if args.type == 'scr':
                    cv2.imwrite(os.path.join(directory + '\\face_found\\', text + '_' + str(td).replace(':', '-') + '.jpg'), image)
                else:
                    cv2.imwrite(os.path.join(directory + '\\face_found\\', os.path.splitext(os.path.basename(args.input))[0] + '_' + text + '_' + str(td).replace(':', '-') + '.jpg'), image)

        # 画像を表示する
        cv2.namedWindow("face recognition", cv2.WINDOW_NORMAL)
        cv2.imshow("face recognition", image)
        if args.type != 'mov' and args.type != 'movie' and args.type != 'cam' and args.type != 'camera':#謎？なんでいれたのか忘れた
            cv2.imwrite(os.path.join(directory + '\\face_found\\', os.path.splitext(os.path.basename(args.input))[0] + id + "_recognized.jpg"), image)

    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()