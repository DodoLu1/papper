# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pandas as pd
import numpy as np
import glob


try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../bin/python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + \
            dir_path + '/../x64/Release;' + dir_path + '/../bin;'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", default="../examples/media/COCO_val2014_000000000192.jpg")
    # help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # parser.add_argument("--video_path", default="../examples/media/testvideo.mp4")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1:
            next_item = args[1][i+1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params:
                params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    # imageToProcess = cv2.imread(
    #    "../examples/media/COCO_val2014_000000000192.jpg")  # args[0].image_path
    imageToProcess = cv2.imread(
        "D:/openpose_data/UT_Kinect/RGB/s01_e01/colorImg252.jpg")
    # 只取walk 動作圖片
    # imagelist = glob.glob(os.path.join(
    #    "D:/openpose_data/UT_Kinect/RGB/s01_e01", "*[g][2-3][0-9][0-9].jpg"))
    # imagelist = imagelist[18:62]  # [18:62]
    # 取全部圖片
    '''
    path = 's01_e02'
    imagelist_1 = sorted(glob.glob(
        'D:/openpose_data/UT_Kinect/RGB/'+path+'/colorImg[0-9][0-9][0-9].jpg'))
    imagelist_2 = sorted(glob.glob(
        'D:/openpose_data/UT_Kinect/RGB/'+path+'/colorImg[0-9][0-9][0-9][0-9].jpg'))
    imagelist = imagelist_1+imagelist_2
    '''

    name = '額外測試'  # 測試
    imagelist = sorted(
        glob.glob('D:/openpose_data/揮手測試/'+name+'/圖檔/' + '圖檔0000*.jpg'))  # 圖檔0000
    output_folder = ('D:/openpose_data/揮手測試/'+name+'/openpose圖片/')
    nopeople = 0
    for img in imagelist:
        imageToProcess = cv2.imread(img)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        x = datum.poseKeypoints
        x = np.array(x)

        sk = []

        if(x.any() == True):
            y = datum.poseKeypoints.shape[0]
            print('人數:\n', y)
            for i in range(y):
                print('person:', i, "Body keypoints:\n", x[i])
                l = x[i].tolist()
                if(i == 0):
                    sk = sk + l
                else:
                    sk = np.append(sk, l)

            sk_0 = np.reshape(sk, (y, 75))
            sk_data = pd.DataFrame(sk_0)
            # 把關節點位儲存到csv
            # sk_data.to_csv('D:/openpose_data/揮手測試/'+name+'/骨架資訊_all.csv',
            #                index=False, mode='a+', header=False)
            # cv2.imwrite(output_folder+img[37:-4]+'.jpg',
            #             datum.cvOutputData)  # img[33:43]

            cv2.namedWindow("OpenPose 1.7.0 - Tutorial Python API",
                            cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OpenPose 1.7.0 - Tutorial Python API", 600, 1000)
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API",
                       datum.cvOutputData)
            cv2.waitKey(0)

        else:
            print('沒有偵測到人')
            nopeople = nopeople+1
    print('沒偵測到:', nopeople)

    # sk_0 = sk.reshape(y, 25, 3)

    # print("sk測試: \n", sk_0)
    # cv2.namedWindow("OpenPose 1.7.0 - Tutorial Python API",
    #                cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("OpenPose 1.7.0 - Tutorial Python API", 800, 600)
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)


except Exception as e:
    print(e)
    sys.exit(-1)
