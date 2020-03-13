# -*- coding: utf-8 -*-

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
import imutils
from datetime import datetime


class ParsingCond:
	def __init__( self, color_no, thrs, adaptive = False):
		self.color_no = color_no
		self.thrs = thrs
		self.adaptive = adaptive
	def __str__(self):
		return "<color_no={},thrs={},is_adaptive={}>".format(self.color_no, self.thrs, self.adaptive)

class Rect:
	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
	def list(self):
		return [[self.x, self.y], [self.x, self.y+self.h], [self.x+self.w, self.y+self.h], [self.x+self.w, self.y]]
	def __str__(self):
		return "<{},{} ~ {},{}>".format(self.x, self.y,  self.w, self.h)

def getIntersectionRatioOfImg(frame_r, img_r):
	if img_r.x > frame_r.x+frame_r.w:
		return 0
	if img_r.x+img_r.w < frame_r.x:
		return 0
	if img_r.y > frame_r.y+frame_r.h:
		return 0
	if img_r.y+img_r.h < frame_r.y:
		return 0

	# 겹친공간
	x = max(img_r.x, frame_r.x);
	y = max(img_r.y, frame_r.y);
	w = min(img_r.x+img_r.w, frame_r.x+frame_r.w) - x;
	h = min(img_r.y+img_r.h, frame_r.y+frame_r.h) - y;

	return ((w * h)/(img_r.w * img_r.h))

def isRevImage(img):
	count = dict()
	for r in range(0, len(img)):
		for c in range(0, len(img[r])):
			v = img[r][c]
			if v not in count:
				count[v] = 0
			count[v] = count[v] + 1
	if  len(count) < 2:
		return False
	scount = sorted(count.items(), key=lambda t : t[1],  reverse=True)
	return(scount[0][0] > scount[1][0])

def getTheshBinary(plate_img, thrs, h):
	THRESH_TYPE = cv.THRESH_BINARY
	plate_img = cv.fastNlMeansDenoising(plate_img, h=int(len(plate_img)*h))
	ret, plate_img = cv.threshold(plate_img, thrs, 255, THRESH_TYPE)
	if isRevImage(plate_img):
		plate_img = cv.bitwise_not(plate_img)
	return plate_img

def findSquaresImg(p_img, debug=0):
	try_parsing_cond_list = [
		ParsingCond(1, 100, False)
		,ParsingCond(1, 128, True)
		,ParsingCond(1, 140, False)
		,ParsingCond(2, 100, False)
		,ParsingCond(0, 100, False)		]
	cont_list = []
	img_list = []
	recct_draw_img = p_img.copy()
	for  pcond in try_parsing_cond_list:
		log(2, 'findSquaresImg>begin')
		bin = cv.split(p_img)[pcond.color_no]
		log(2, 'findSquaresImg>cv.threshold.begin')
		if pcond.adaptive:
			bin = cv.adaptiveThreshold(bin, pcond.thrs, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,4)
		else:
			_retval, bin = cv.threshold(bin, pcond.thrs, 255, cv.THRESH_BINARY)
		log(2, 'findSquaresImg>cv.threshold.end')
		imshow(3, 'findSquaresImg threshold', bin, 0, 400, 800, is_wait=True)
		log(2, 'findSquaresImg>cv.findContours.begin')
		contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
		log(2, 'findSquaresImg>cv.findContours.end')

		find_count = 0
		for cont in contours:
			cont_len = cv.arcLength(cont, True)
			cont = cv.approxPolyDP(cont, 0.02*cont_len, True)
			if  len(cont) == 4 and cv.contourArea(cont) > 2000 and  cv.isContourConvex(cont):
				recct_draw_img = cv.drawContours(recct_draw_img, [cont], -1, (0, 0, 255), 2)
			if len(cont) == 4 and cv.contourArea(cont) > 2000 and cv.isContourConvex(cont):
				cont = cont.reshape(-1, 2)
				if  isHorizontalRectangle(cont):
					#수평 직사각형으로 
					img_list.append(subImage2Horizontal(p_img, cont))
					cont_list.append(cont)
					recct_draw_img = cv.drawContours(recct_draw_img, [cont], -1, (0, 255, 0), 2)
					find_count = find_count + 1
					
		log(2, 'PCOND>{}-{}'.format(pcond, find_count))
	#img_list = sorted(img_list, key=lambda img: len(img)*len(img[0]), reverse=True)

	for img in img_list:
		log(2, 'findSquaresImg {}'.format(len(img[0])))
	return img_list, recct_draw_img, cont_list

# 측면에서 찍어 기울진 영상을 수평 직사각형으로 변형한다.
def subImage2Horizontal(img, p1):
	s = sortLeftTopRightDown(p1)

	h1 = (int) ((  (s[1][1] - s[0][1]) + (s[3][1] - s[2][1])) / 2)
	w1 = (int) ((  (s[2][0] - s[0][0]) + (s[3][0] - s[1][0])) / 2)
	pts1 = np.float32([s[0], s[1], s[2], s[3]])
	pts2 = np.float32([[0,0],[0,h1],[w1,0],[w1,h1]])
	transformMatrix = cv.getPerspectiveTransform(pts1,  pts2)
	img = cv.warpPerspective(img, transformMatrix, (w1,h1))
	return img

# cv.warpPerspective 처리전 위치 역으로 계산하기 
def getOrgPostion(cont, w, h, px, py):
	s = sortLeftTopRightDown(cont)
	pxx = px/w
	pyy = py/h	
	return int(s[0][0] + (s[3][0] - s[0][0]) * pxx +0.5),  int(s[0][1] + (s[3][1] - s[0][1]) * pyy +0.5)

def isHorizontalRectangle(s):
	if len(s) != 4:
		return False
	s = sortLeftTopRightDown(s)
	# 좌표점은 좌상->좌하->우상->우하
	hl = getDistance(s[0], s[1])
	hr = getDistance(s[2], s[3])
	wu = getDistance(s[0], s[2])
	wd = getDistance(s[1], s[3])

	minchar_size = 20
	if  minchar_size*6 < wu and minchar_size*6 < wd and minchar_size < hl and minchar_size < hr:
		if absRatio(wu,wd) < 0.2 and absRatio(hl,hr) < 0.2:
			if   1.2 < (wu/hl) and  (wu/hl) < 10:
				return True
	return False

def getDistance(p1, p2):
	x = p1[0] - p2[0]
	y = p1[1] - p2[1]
	return (int)(math.sqrt(x*x + y*y))

def absRatio(p1, p2):
	if p1 == 0 or p2 == 0:
		return 99
	return abs(p1 - p2) / min(p1, p2)

# 좌표점은 좌상->좌하->우상->우하
def sortLeftTopRightDown(plist):
	sortx = sorted(plist,  key=lambda p: p[0])
	l = sorted(sortx[0:2].copy(), key=lambda p: p[1])
	r = sorted(sortx[2:4].copy(), key=lambda p: p[1])
	rlist = plist.copy()
	rlist[0] = l[0]
	rlist[1] = l[1]
	rlist[2] = r[0]
	rlist[3] = r[1]
	return  rlist

def getDiagonal(x, y):
	return (int)(sqrt(x*x + y*y))

def loadDefImg(train_file):
	img_dic = {}
	with open (train_file, "r") as fp:
		for line in fp.readlines():
			line = line.replace('\n', '')
			if len(line) >= 3 and not line.startswith( '#' ):
				c = line.split( '\t' )
				if len(c) >= 2:
					img_dic[c[0]] = c[1]
	return img_dic

def loadCharSet(train_file):
	# text 파일에서 # 코멘트를 제외하고 unique char list
	rbuf = []
	with open (train_file, "r") as fp:
		for l in fp.readlines():
			if not l.startswith( '#' ):
				rbuf.append(l)
	string_text =  "".join(rbuf).replace('\n', '').replace(' ', '').replace('	', '')
	char_list = list(string_text)
	char_list = list(set(char_list))
	char_list.sort()
	char_dict = { i : char_list[i] for i in range(0, len(char_list) ) }
	return char_dict

font_dict = {}
def genImage(c, font_file, r, gab=1):
	font = font_dict.get(font_file, None)
	if font == None:
		font = ImageFont.truetype(font_file, 96)
		font_dict[font_file] = font
	img = np.zeros((128,128), np.uint8)
	img_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(img_pil)
	draw.text((gab, gab),  c, font = font, fill = (255))
	img_pi = img_pil.rotate(r)
	return np.asarray(img_pi)

def genImage28(c, font_file, r, gab=1):
	img_pi = genImage(c, font_file, r, gab=1)
	img_fit = img28Fit(img_pi, gab)
	return img_fit

def img28Fit(img, p_gab_x, p_gab_y=None, is_rev=False):
	if p_gab_y == None:
		p_gab_y = p_gab_x
	h = len(img)
	w = len(img[0])
	retval, bin = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

	# 문자위치 희색 배경이라 전체가 글자로 보임으로 반전하여(검정바탕 흰글씨) 글자 위치을 찾는다
	#imshow('img28Fit', bin, 0, 0)
	bin1 = bin
	if is_rev:
		bin1 = cv.bitwise_not(bin)
	contours, _hierarchy = cv.findContours(bin1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	if len(contours) > 0:
		rect1 = boundingRect(contours)
		gab_y = int(p_gab_y*(rect1.h/28))
		gab_x = int(p_gab_x*(rect1.w/28))
		#print(rect1, len(bin), len(bin[0]))
		r = extRect(rect1, len(img[0]), len(img), gab_x, gab_y)
		m = max(r.w, r.h)
		bin_1 = bin[r.y:r.y+m, r.x:r.x+m]
		#imshowAndCommand('img28Fit_1', bin_1, 0, 300)
		bin = bin_1
	bin = cv.resize(bin, (28, 28))

	return bin

def extRect(r, w, h, gab_x=0, gab_y=0):
	if r.x > gab_x :
		r.x -= gab_x
	if r.w > gab_x :
		r.w += gab_x

	if r.y > gab_y :
		r.y -= gab_y
	if r.h > gab_y :
		r.h += gab_y

	if r.x + r.w  + gab_x < w:
		r.w += gab_x
	if r.y + r.h  + gab_y < h:
		r.h += gab_y
	return r

def copy2d(dst, src):
	for r in range(min(len(dst), len(src))):
		for c in range(min(len(dst[r]), len(src[r]))):
			dst[r][c] = src[r][c]

def imgToArray(img_pil):
	return np.asarray(img_pil, dtype="float" ).reshape(-1)

def boundingRect(ctrs):
		r = Rect(9999, 9999, 0, 0)
		for ctr in ctrs:
			x, y, w, h = cv.boundingRect(ctr)
			r.x = min(r.x, x)
			r.y = min(r.y, y)
			r.w = max(r.w, w + x)
			r.h = max(r.h, h + y)
		#w, h를 절 때 위치에서 크기로 변경
		r.w = r.w - r.x
		r.h = r.h - r.y
		return r

def findValueDic(dic, val):
	for c in dic.keys():
		if dic[c] == val:
			return c
	return None

def charFormat(str):
	if str == None or len(str) == 0:
		return str

	ret = ''
	for c in range(len(str)):
		if '0'  <= str[c]  and str[c] <= '9':
			ret += 'n'
		elif isHangul(str[c]):
			ret += 'h'
		else:
			ret += 'e'
	return ret

def isHangul(c):
	if 0xac00 <= ord(c) and ord(c) <= 0xd7a3:
		return True
	return False

def subImageByRect(img, r):
	x  = int(r.x)
	y  = int(r.y)
	x2 = int(r.x + r.w)
	y2 = int(r.y + r.h)
	return img[y:y2, x:x2]

def drawRect(img, r, color, linew):
	cv.rectangle(img,(r.x,r.y),( r.x + r.w, r.y + r.h), color, linew)

def drawRects(img, rect_list,  color, linew):
	for r in rect_list:
		drawRect(img, r, color, linew)

def pasteImage(back, fore, x, y):
	ret = back.copy()
	for r in range(0, len(fore)):
		for c in range(0, len(fore[r])):
			ret[r+y][c+x] = fore[r][c]
	return ret

def grayScale(img, thrs=80, color_no=2):
	#흑백반전
	try:
		img1 = cv.split(img)[color_no]
	except:
		img1 = img
	retval, bin = cv.threshold(img1, thrs, 255, cv.THRESH_BINARY)
	return bin

def revImage(img, thrs=80, color_no=2):
	#흑백반전
	img = cv.bitwise_not(img)
	return grayScale(img, thrs, color_no)

def revImage2(img, thrs=80):
	#흑백반전
	img = cv.bitwise_not(img)
	retval, bin = cv.threshold(img, thrs, 255, cv.THRESH_BINARY)
	return bin

def extractCharImage(bin):
	# 문자만 추줄
	contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	r = boundingRect(contours)
	bin = subImageByRect(bin, r)
	return bin

def rotateImage(bin, ro):
	#기울기 변경
	bin = cv.bitwise_not(bin)
	bin = imutils.rotate(bin, ro)
	bin = cv.bitwise_not(bin)
	return bin

def minmaxImage(bin, g=3):
	#이미 작게 했다 크게 하기
	h = len(bin)
	w = len(bin[0])
	bin = cv.resize(bin, (int(w/g), int(h/g)))
	bin = cv.resize(bin, (w, h))
	return bin

def extendImage(bin, g=3, is_square=False):
	#3 큰 배경에 복사해 넣기
	h = len(bin)
	w = len(bin[0])
	if is_square:
		m = max(h, w)
		h = w = m
	p = g/2-0.5
	blank_a = np.zeros((h*g, w*g), np.uint8)
	blank_a.fill(255)
	bin = pasteImage(blank_a, bin, int(p*w), int(p*h))
	return bin

def SqueezeCharImage(bin, ulu = 0, ull = 0, dld = 0, dll = 0, drd = 0, drr = 0, urr = 0, uru = 0, is_rev=False):
	#imshowAndCommand('SqueezeCharImage', bin, 0, 0)
	# 문자위치 희색 배경이라 전체가 글자로 보임으로 반전 하여 글자 위치을 찾는다
	bin1 = bin
	if is_rev:
		bin1 = cv.bitwise_not(bin)
	contours, _hierarchy = cv.findContours(bin1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	r = boundingRect(contours)
	#찌그리기 u/d, l/r, 방항[udlr]
	ulu *= r.h
	ull *= r.w
	dld *= r.h
	dll *= r.w
	drd *= r.h
	drr *= r.w
	urr *= r.w
	uru *= r.h
	pts1 = np.float32([[r.x-ull, r.y-ulu], [r.x-dll, r.y+r.h+dld], [r.x+r.w+drr, r.y+r.h+drd], [r.x+r.w+urr,r.y-uru]])
	pts2 = np.float32([[0,0],[0,r.h],[r.w,r.h],[r.w,0]])
	transformMatrix = cv.getPerspectiveTransform(pts1,  pts2)
	bin = cv.warpPerspective(bin, transformMatrix, (r.w, r.h), borderMode=cv.BORDER_TRANSPARENT)

	return bin

def splitextPathExt(inputFilepath):
	filename_w_ext = os.path.basename(inputFilepath)
	filename, file_extension = os.path.splitext(filename_w_ext)
	path, filename = os.path.split(inputFilepath)
	filename = filename.replace(file_extension, '')
	return path, filename, file_extension

def getNow():
	return  datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

set_debug_level = 0
def setDebugLevel(debug):
	global set_debug_level
	set_debug_level = debug

def log(debug, s):
	global set_debug_level
	if debug > set_debug_level :
		return
	print(getNow(), s)

def imshow(debug, title, img, x=0, y=0 , w=None, h=None, is_wait=False):
	global set_debug_level
	if debug > set_debug_level :
		return
	img_1 = img
	if w != None and h == None:
		h1 = img.shape[0]
		w1 = img.shape[1]
		if w < w1:
			img_1 = cv.resize(img, (w, int(w * (h1/w1)) ))
	elif w != None and h != None:
		img_1 = cv.resize(img, (w,h))
	cv.imshow(title, img_1)
	cv.moveWindow(title, x,y)
	cv.imwrite("tmp/{}.jpg".format(title), img_1)
	if not is_wait:
		ch = cv.waitKey(1)
	else:
		print("wait key ....", title)
		ch = cv.waitKey(0)
		if ch == 27:
			print("esc exit...");
			sys.exit(0)
		if ch == ord('s') :
			cv.imwrite("s.jpg", img)
			print("img write ...s.jpg")

def imshowAndCommand(pr_level, title, img, x=0, y=0 , w=None, h=None):
	global set_debug_level
	if pr_level > set_debug_level :
		return
	imshow(pr_level, title, img, x, y, w, h, True)
