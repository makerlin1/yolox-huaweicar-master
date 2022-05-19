import os
import cv2
from functools import reduce
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import argparse

class image_bnk():
    def __init__(self, path):
        self.file_name_list = np.array([os.path.join(path, p) for p in os.listdir(path) if p[-1] == 'g'])
        self.memory = self.store_img_vec()

    def store_img_vec(self):
        print("store the image vec")
        amount = len(self.file_name_list)
        orb = cv2.ORB_create()
        memory = []
        for i in tqdm(range(amount)):
            img1_path = self.file_name_list[i]
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) 
            _, des1 = orb.detectAndCompute(img1, None) #(500,32)
            memory.append(des1.reshape(1,500,32))
        return np.vstack(memory)

    def __call__(self, threshold):
        print("Begining")
        result_log = {}
        while self.memory.shape[0] != 0:
            print("wait for {} images to process...".format(self.memory.shape[0]))
            query = self.memory[0]
            index = []
            result_log[self.file_name_list[0]] = []

            for i in range(self.memory.shape[0]):
                simi = self.similarity(query, self.memory[i])
                if simi > threshold:
                    index.append(i)
                    result_log[self.file_name_list[0]].append(self.file_name_list[i])

            self.memory = np.delete(self.memory, index, axis=0)
            self.file_name_list = np.delete(self.file_name_list, index, axis=0)
        with open("result.json",'w') as f:
            json.dump(result_log,f)
        
    def similarity(self, vec1, vec2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # knn筛选结果
        matches = bf.knnMatch(vec1, trainDescriptors=vec2, k=2)
        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        similary = len(good) / len(matches)
        return similary



parser = argparse.ArgumentParser("filter the same image")
parser.add_argument('-fp',help="the folder path")
parser.add_argument('-thres', type= float,help='threshold of similarity')
args = parser.parse_args()
filter = image_bnk(args.fp)
filter(args.thres)
