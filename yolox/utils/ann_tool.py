import cv2
import os
import argparse
parser = argparse.ArgumentParser("split the video")
parser.add_argument('-fp',help="the folder path to save new image")
parser.add_argument('-path', help='path of video')
args = parser.parse_args()
cap = cv2.VideoCapture(args.path)
num = 0
count = 0
framerate = int(cap.get(5) / 5)
while cap.isOpened():
    count += 1
    ret,frame = cap.read()
    cv2.imshow(args.path, frame) #显示画面
    if count % framerate != 0:
        continue 
    key = cv2.waitKey(24)
    while key != ord('w') and key != ord("s"):
        key = cv2.waitKey(24) 
    if key == ord("s"):
        save_image_dir = os.path.join(args.fp, "test-{}.jpg".format(num))
        num += 1
        cv2.imwrite(save_image_dir,frame) 
cap.release()         #释放摄像头
cv2.destroyAllWindows() #释放所有显示图像窗口
 

