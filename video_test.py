# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

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
parser.add_argument("--image_path", default="../examples/media/test.jpg")
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
#oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()
# imageToProcess = cv2.imread(
#    "../examples/media/test.jpg")  # args[0].image_path
# videoToProcess = cv2.VideoCapture(
#    "../examples/media/video.avi")  # testvideo.mp4 #video.avi
videoToProcess = cv2.VideoCapture(
    "D:/openpose_data/揮手.MP4")
# datum.cvInputData = imageToProcess
# opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# VIDEO
sk = [[]for i in range(10)]

while(videoToProcess.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = videoToProcess.read()
    if ret == True:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        x = datum.poseKeypoints
        x = np.array(x)
        if(x.any() == True):
            y = datum.poseKeypoints.shape[0]
            print('人數:\n', y)
            for i in range(y):
                print('person:', i, "，Body keypoints:\n", x[i])
                l = x[i].tolist()
                sk[i] = sk[i]+l

        else:
            print('沒有偵測到人')
        #print("Body keypoints: \n" + str(datum.poseKeypoints))

        #cv2.namedWindow('Frame',  cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', datum.cvOutputData)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

# Release the video capture object
videoToProcess.release()
cv2.destroyAllWindows()
sz_0 = len(sk[0])
#sk_n = np.array(sk[0])
sk_0 = np.reshape(sk[0], (int(sz_0/25), 25, 3))
print('sk測試person1: \n')
print(sk_0[0])
print(sz_0/25)

sz_1 = len(sk[1])
sk_1 = np.reshape(sk[1], (int(sz_1/25), 25, 3))
print('sk測試person2: \n')
print(sk_1[0])
print(sz_1/25)

sz_2 = len(sk[2])
sk_2 = np.reshape(sk[2], (int(sz_2/25), 25, 3))
print('sk測試person2: \n')
print(sk_2[0])
print(sz_2/25)

sz_3 = len(sk[3])
sk_3 = np.reshape(sk[3], (int(sz_3/25), 25, 3))
print('sk測試person2: \n')
print(sk_3[0])
print(sz_3/25)

sz_4 = len(sk[4])
sk_4 = np.reshape(sk[4], (int(sz_4/25), 25, 3))
print('sk測試person2: \n')
print(sk_4[0])
print(sz_4/25)


sz_5 = len(sk[5])
sk_5 = np.reshape(sk[5], (int(sz_5/25), 25, 3))
print('sk測試person2: \n')
print(sk_5[0])
print(sz_5/25)


print(sk[0], '\n分隔\n', sk[1])
np.savetxt("numpy_test_0.csv", sk[0], delimiter=",", fmt='% s')
np.savetxt("numpy_test_1.csv", sk[1], delimiter=",", fmt='% s')
# Display Image

# print("Body keypoints: \n" + str(datum.poseKeypoints))

# cv2.namedWindow("OpenPose 1.7.0 - Tutorial Python API",  cv2.WINDOW_NORMAL)
# cv2.resizeWindow("OpenPose 1.7.0 - Tutorial Python API", 800, 600)
# cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
# cv2.waitKey(0)
