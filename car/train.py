# -*- coding: utf-8 -*-

# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import argparse
import sys
import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import glob
import math
from skimage import util

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import Cstool
import Dnn28
import pytesseract

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
ttf_path = "ttf/*.ttf"
dnn28 = None
char_dict=[]

##################################################################################################################
def usage_exit():
	print("python train.py train_id repeat_count [-c] ")
	print("       #train_id directory Directory별 내용을 읽어 하후 train_id로 된 결과 남김")
	print("       #repeat_count 만큼 반복 학습 ")
	print("       #-c 지금까지 학습내용을 지우고 다시 학습")
	sys.exit(0)

if __name__ == '__main__':
	if len(sys.argv) < 3:
		usage_exit()

	train_id = sys.argv[1]
	repeat_count = int(sys.argv[2])

	if not os.path.isdir("./"+train_id):
		print("{}로된 subdirectory가 없습니다".format(train_id))
		sys.exit(0)

	# train_id.txt로된 char set 읽기
	char_dict = Cstool.loadCharSet("dnn/"+train_id +".txt")
	print("dictionary  {}".format(char_dict))
	print("dictionary loading {}".format(len(char_dict)))
	nclass = len(char_dict)
	dnn = Dnn28.Dnn28(nclass, train_id)

	if len(sys.argv) < 4 or sys.argv[3] != "-c":
		dnn.restore()
	images = []
	labels = []
	for c  in range(nclass):
		ch = char_dict[c]
		path="{}/{}/".format(train_id, ch)
		file_list = os.listdir(path)
		for fname in file_list:
			#print("FNAME {}".format(os.path.join(path, fname) ))
			img = cv.imread(os.path.join(path, fname), cv.IMREAD_UNCHANGED)
			img28 = cv.resize(img, (28, 28))
			is_rev = Cstool.isRevImage(img28)
			if is_rev:
				img28  = Cstool.revImage2(img28, thrs=120)
			Cstool.imshowAndCommand(9, 'img28', img28, 0, 0)
			imgarr = Cstool.imgToArray(img28)
			images.append(imgarr)
			label = np.zeros(nclass, np.uint8)
			label[ c ] = 1
			labels.append(label)
		print(".", end="", flush=True)
	print("image loading {} ok".format(len(labels)))

	print("start train".format(len(labels)))
	batch_size = 50
	for i in range(repeat_count):
		if i != 0 and i % 10 == 00:
			print("save..")
			dnn.save()
			print("save..ok")
			print('{} training accuracy ==>'.format(i))
			for ind in range(0, len(images), batch_size):
				batch_images = images[ind:ind+batch_size]
				batch_labels = labels[ind:ind+batch_size]
				accuracy = dnn.evalAccuracy(batch_images, batch_labels)
				print('{:04.2f} '.format(accuracy[0]), end="",flush=True)
			print("")
		# Batch Size 50씩 잘라 처리
		for ind in range(0, len(images), batch_size):
			batch_images = images[ind:ind+batch_size]
			batch_labels = labels[ind:ind+batch_size]
			dnn.train(batch_images, batch_labels)
		#print(".", end="", flush=True)
		#Cstool.imshowAndCommand('batch_images={}'.format(batch_labels[0]), batch_images[0], 0, 0)
	print("save..")
	dnn.save()
	print("save..ok")
