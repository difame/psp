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
adjust_plate_width_max = 400

class ParsingCharCond:
	def __init__( self, color_no, thrs, h):
		self.color_no = color_no
		self.thrs = thrs
		self.h = h
	def __str__(self):
		return "<color_no={},thrs={},h={}>".format(self.color_no, self.thrs, self.h)

class CarnoFrame:
	char_dict = []
	dnn = None
	def __init__( self, title, width, height, char_format, char_rect_list):
		self.title = title
		self.width = width
		self.height = height
		self.char_format = char_format
		self.char_rect_list = char_rect_list

	def calcRectList(self, plate_img_width, plate_img_height):
		w_ratio = plate_img_width /self.width
		h_ratio = plate_img_height/self.height

		calc_char_rect_list = []
		for p in self.char_rect_list:
			calc_char_rect_list.append(Cstool.Rect(
				(int)(p.x * w_ratio),
				(int)(p.y * h_ratio),
				(int)(p.w * w_ratio),
				(int)(p.h * h_ratio)))
		return calc_char_rect_list

	def findCarnoInPlate(self, plate_img, easy):
		char_rect_list = []

		plate_img_height= plate_img.shape[0]
		plate_img_width = plate_img.shape[1]
		no = 0
		car_no_text = ""
		recalc_rect_list = self.calcRectList(plate_img_width, plate_img_height)

		ctrs, hier = cv.findContours(plate_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		Cstool.imshowAndCommand(4, 'findCarnoInPlate im2>' + self.title, plate_img, 0, 500)
		Cstool.log(4, "findCarnoInPlate() {} ({},{}) ctrl={}<<<<<<<<".format(self.title, plate_img_width, plate_img_height, len(ctrs)))
		if Carno.debug >= 4:
			plate_img_with_rect = plate_img.copy()
			Cstool.drawRects(plate_img_with_rect, recalc_rect_list, (128), 3)
			Cstool.drawRects(plate_img_with_rect, recalc_rect_list, (0),   2)
			Cstool.drawRects(plate_img_with_rect, recalc_rect_list, (255), 1)
			Cstool.imshowAndCommand(4, 'findCarnoInPlate FORM>' + self.title, plate_img_with_rect, 0, 500)

		for plate_char_rect in recalc_rect_list:
			# 예상문자영역에 60%이상 겹친 문자조각(ctrs)만 추출한다.
			in_ctrs = self.getCtrs(plate_char_rect, ctrs, 0.6)
			Cstool.log(4, "findCarnoInPlate() {} getCtrs ==> ctrs count={}".format(plate_char_rect, len(in_ctrs) ))
			if len(in_ctrs) == 0:
				return None, None

			ctrs_rect1 = self.boundingRect( in_ctrs )
			if  (Cstool.getIntersectionRatioOfImg(frame_r=ctrs_rect1, img_r=plate_char_rect) < 0.3) or \
				(Cstool.getIntersectionRatioOfImg(frame_r=plate_char_rect, img_r=ctrs_rect1) < 0.5) :
				Cstool.log(5, "findCarnoInPlate() exit not found getIntersectionRatio()")
				return None, None

			ctrs_rect = Cstool.extRect(ctrs_rect1, plate_img_width, plate_img_height)
			ch_img = plate_img[ctrs_rect.y:ctrs_rect.y+ctrs_rect.h, ctrs_rect.x:ctrs_rect.x+ctrs_rect.w]

			ch, dnn_img = self.recognizeChar28(ch_img)
			Cstool.log(4, 'recognizeChar28>>>' +ch)
			Cstool.imshow(4, 'recognizeChar28', ch_img, 1200, 0)
			Cstool.imshowAndCommand(4, 'findCarnoInPlate dnn_img', dnn_img, 1200, 400)

			if ch == None:
				Cstool.log(5, "findCarno() exit not found recognizeChar28()")
				return None, None
			car_no_text = car_no_text + ch

			Cstool.imshow(4, "dnn_img{}".format(no, ch), dnn_img, 800, 120*no)
			Cstool.log(3, "dnn_img{}==>{}".format(no, ch))
			no += 1
			char_rect_list.append([ch, ctrs_rect1])

		find_no_format = Cstool.charFormat(car_no_text)
		if find_no_format == self.char_format or (easy and "nnnn" in find_no_format) :
			if Carno.debug >= 2:
				Cstool.log(2, 'findCarnoInPlate OK =[{}]'.format(car_no_text))
				plate_img_with_rect = plate_img.copy()
				Cstool.drawRects(plate_img_with_rect, recalc_rect_list, (255), 3)
				Cstool.imshowAndCommand(3, 'findCarnoInPlate OK.>' + self.title, plate_img_with_rect, 0, 500)
			return car_no_text, char_rect_list
		return None, None

	def boundingRect(self,  ctrs ):
		r = Cstool.Rect(9999, 9999, 0, 0)
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

	# 예상 문자영역(plate_rect)에 정해진 비울Dnn28이상 들어간 이미지조작(ctrs)를 찾는다
	def getCtrs(self, plate_rect, ctrs, base_ratio):
		in_ctrs = []
		for ctr in ctrs:
			x, y, w, h = cv.boundingRect(ctr)
			# 문자조각이 너무 작거나 크면 무시
			if  (w < plate_rect.w*0.2 and h < plate_rect.h*0.2) or plate_rect.w*1.2  < w or plate_rect.h*1.2 < h :
				continue
			# 예상문자 위치와 약간 어긋 나도 수용 한다
			ctr_rect = Cstool.Rect(x,y,w,h)
			ratio = Cstool.getIntersectionRatioOfImg(frame_r=plate_rect, img_r=ctr_rect)
			if ratio > base_ratio:
				in_ctrs.append(ctr)
		return in_ctrs


	def recognizeChar28(self, img):
		img28 = cv.resize(img, (28, 28))
		imgarr = Cstool.imgToArray(img28)
		output = CarnoFrame.dnn.evalOutput(imgarr)
		return CarnoFrame.char_dict[output[0]], img28

class Carno:
	debug = 0
	def __init__(self, train_id, p_debug=0):
		Carno.debug = p_debug
		CarnoFrame.char_dict = Cstool.loadCharSet('car/dnn/'+train_id +".txt")
		CarnoFrame.dnn = Dnn28.Dnn28(len(CarnoFrame.char_dict), train_id)
		CarnoFrame.dnn.restore()

		a = 7
		b = 12
		c = 10
		d = 7
		self.carno_frame_list = dict()
		self.carno_frame_list["nor_wide"] = CarnoFrame("nor_wide", 520, 110, "nnhnnnn",
		[	Cstool.Rect(44+56*0,    13, 56, 83),#13.5
			Cstool.Rect(44+56*1,     13, 56, 83),#13.5
			Cstool.Rect(44+56*2,     13, 60, 83),#13.5
			Cstool.Rect(44+56*2+96,  13, 56, 83),#13.5
			Cstool.Rect(44+56*3+96,  13, 56, 83),#13.5
			Cstool.Rect(44+56*4+96,  13, 56, 83),#13.5
			Cstool.Rect(44+56*5+96,  13, 56, 83)])#13.5
		self.carno_frame_list["nor_box"] = CarnoFrame("nor_box", 335, 155, "nnhnnnn",
		[	Cstool.Rect(4+45*0,     46, 45, 83),#46.2
			Cstool.Rect(4+45*1,     46, 45, 83),#46.2
			Cstool.Rect(4+45*2,     46, 49, 83),#46.2
			Cstool.Rect(4+45*2+49,  46, 45, 83),#46.2
			Cstool.Rect(4+45*3+49,  46, 45, 83),#46.2
			Cstool.Rect(4+45*4+49,  46, 45, 83),#46.2
			Cstool.Rect(4+45*5+49,  46, 45, 83)])#46.2
		self.carno_frame_list["per_bigbox"] = CarnoFrame("per_bigbox", 440, 200, "nnhnnnn",
		[	Cstool.Rect(11+59*0,     60, 59, 105),
			Cstool.Rect(11+59*1,     60, 59, 105),
			Cstool.Rect(11+59*2,     60, 64, 105),
			Cstool.Rect(11+59*2+64,  60, 59, 105),
			Cstool.Rect(11+59*3+64,  60, 59, 105),
			Cstool.Rect(11+59*4+64,  60, 59, 105),
			Cstool.Rect(11+59*5+64,  60, 59, 105)])
		self.carno_frame_list["old_blue"] = CarnoFrame("old_blue", 345, 170, "hhnnhnnnn",
		[	Cstool.Rect(92.5+40*0,   20, 30+b, 35),
			Cstool.Rect(92.5+40*1,   20, 30+b, 35),
			Cstool.Rect(92.5+40*2,   20, 30+b, 35),
			Cstool.Rect(92.5+40*3,   20, 30+b, 35),
			Cstool.Rect(25,          70, 35+b, 75),
			Cstool.Rect(100+55*0,    70, 45+c, 75),
			Cstool.Rect(100+55*1,    70, 45+c, 75),
			Cstool.Rect(100+55*2,    70, 45+c, 75),
			Cstool.Rect(100+55*3,    70, 45+c, 75)])
		self.carno_frame_list["old_blue"] = CarnoFrame("old_blue1", 315, 160, "nnhnnnn",
		[	Cstool.Rect(87 ,        11, 45, 43),
			Cstool.Rect(138,        11, 45, 43),
			Cstool.Rect(190,        11, 45, 43),
			Cstool.Rect(10+76*0,    67, 65, 87),
			Cstool.Rect(10+76*1,    67, 65, 87),
			Cstool.Rect(10+76*2,    67, 65, 87),
			Cstool.Rect(10+76*3,    67, 65, 87)])
		self.carno_frame_list["biz_wide"] = CarnoFrame("biz_wide", 520, 110, "hhnnhnnnn",
		[	Cstool.Rect(32+55*0,     10, 55, 41),#13.5, 55, 41.5),
			Cstool.Rect(32+55*0,     55-3, 55, 45),#55,   55, 41.5),
			Cstool.Rect(32+55*1,     13, 55, 83),#13.5
			Cstool.Rect(32+55*2,     13, 55, 83),#13.5
			Cstool.Rect(32+55*3,     13, 55, 83),#13.5
			Cstool.Rect(32+55*3+71,  13, 55, 83),#13.5
			Cstool.Rect(32+55*4+71,  13, 55, 83),#13.5
			Cstool.Rect(32+55*5+71,  13, 55, 83),#13.5
			Cstool.Rect(32+55*6+71,  13, 55, 83)]	)#13.5
		self.carno_frame_list["biz_box"] = CarnoFrame("biz_box", 335, 170, "hhnnhnnnn",
		[	Cstool.Rect(65+17,      9, 40+a, 48),
			Cstool.Rect(65+17+40,   9, 40+a, 48),
			Cstool.Rect(65+17+95,   9, 38, 48),
			Cstool.Rect(65+17+95+38,9, 38, 48),
			Cstool.Rect(9.5,        68, 60+a, 92),
			Cstool.Rect(9.5+60,     68, 62, 92),
			Cstool.Rect(9.5+60+62*1,68, 62, 92),
			Cstool.Rect(9.5+60+62*2,68, 62, 92),
			Cstool.Rect(9.5+60+62*3,68, 62, 92)])
		self.carno_frame_list["biz_bigbox"] = CarnoFrame("biz_bigbox", 440, 200, "hhnnhnnnn",
		[	Cstool.Rect(107,         11, 63+a, 61),
			Cstool.Rect(107+63,      11, 63+a, 61),
			Cstool.Rect(107+126,     11, 50, 61),
			Cstool.Rect(107+126+50,  11, 50, 61),
			Cstool.Rect(20,          84, 89+a, 116),
			Cstool.Rect(20+89,       84, 78, 116),
			Cstool.Rect(20+89+78*1,  84, 78, 116),
			Cstool.Rect(20+89+78*2,  84, 78, 116),
			Cstool.Rect(20+89+78*3,  84, 78, 116)])

	# 번호판 후보 rectange를 찾은후 번호판으로 가정하고, 읽기를 시도한다
	def findCarnoInPicture(self, car_img, file_name="dummy", easy=False):
		Cstool.log(2, 'findCarnoInPicture.begin')
		Cstool.imshow(2, 'findCarnoInPicture', car_img, 0, 0, 800)
		Cstool.log(2, 'findCarnoInPicture>Cstool.findSquaresImg.begin')
		plate_img_list, draw_rect_img, cont_list = Cstool.findSquaresImg(car_img, Carno.debug)

		Cstool.log(3, 'findCarnoInPicture>Cstool.findSquaresImg.end')
		Cstool.imshow(2, 'findCarnoInPicture', draw_rect_img, 0, 0, 800, is_wait=True)
		for i in range(0, len(plate_img_list)):
			plate_img = plate_img_list[i]
			cont = cont_list[i]
			Cstool.log(2, 'findCarnoInPicture>self.recognizeCharInRect.begin')
			car_no, char_rect_list = self.recognizeCharInRect(plate_img, easy)
			Cstool.log(2, 'findCarnoInPicture>self.recognizeCharInRect.end')
			if car_no != None:
				Cstool.log(1, "find_carno={}".format( car_no))
				Cstool.imshow(1, 'findCarnoInPicture', draw_rect_img, 0, 0, 10, is_wait=True)
				return car_no, Carno.VocXml(file_name, len(car_img[0]), len(car_img),
											cont, len(plate_img[0]), len(plate_img), char_rect_list, car_img)

		return None, None

	# 번호판 후보 rectange을 번호판으로 가정하고 인식시도
	def recognizeCharInRect(self, plate_img_color, easy=True):
		try_parsing_cond_list = [
			ParsingCharCond(1, 100, 0.03)
			,ParsingCharCond(1, 180, 0.3)
			,ParsingCharCond(2, 150, 0.1) #청색 오염
			,ParsingCharCond(2, 80, 0.03) #노란색 오염번호판
			,ParsingCharCond(0, 80, 0.03)
			,ParsingCharCond(2, 120, 0.03) #노란색 오염번호판
			,ParsingCharCond(2, 170, 0.03) #노란색 오염번호판
		]

		# 이미지가 너무 크면 줄인다
		h1 = plate_img_color.shape[0]
		w1 = plate_img_color.shape[1]
		#if w1 > adjust_plate_width_max: # 400
		#	Cstool.log(2, 'adjust_plate_width_max {} => {}'.format(w1,adjust_plate_width_max) )
		#	plate_img_color = cv.resize(plate_img_color, (adjust_plate_width_max, int(adjust_plate_width_max * (h1/w1)) ))

		Cstool.imshow(3, 'recognizeCharInRect plate_img_color', plate_img_color, 0, 400,1200, is_wait=True)
		for  pcond in try_parsing_cond_list:
			Cstool.log(2, 'recognizeCharInRect>pcond>{}'.format(pcond))
			plate_bin = cv.split(plate_img_color)[pcond.color_no]
			Cstool.imshow(3, 'recognizeCharInRect plate_bin1', plate_bin, 0, 200, 1200, is_wait=True)
			plate_bin = Cstool.getTheshBinary(plate_bin, pcond.thrs, pcond.h)
			Cstool.imshow(3, 'recognizeCharInRect plate_bin2', plate_bin, 0, 400, 1200, is_wait=True)
			
			for carno_frame in self.carno_frame_list.values():
				car_no_text,  char_rect_list = carno_frame.findCarnoInPlate(plate_bin, easy)
				if car_no_text != None:
					Cstool.imshow(2, 'recognizeCharInRect plate_bin', plate_bin, 0, 800,1200, is_wait=True)
					return car_no_text, char_rect_list
		return None,None

	def findProcess(self, img_fname, easy=True):
		Cstool.log(2,"find carno start")
		if not os.path.isfile(img_fname) :
			return "{} 파일이 없습니다".format(img_fname), None
		car_img = cv.imread(img_fname, cv.IMREAD_UNCHANGED)
		try :
			Cstool.log(2,"image {} load..{}".format(img_fname, len(car_img) ))
		except :
			return "{} 파일을 읽을 수 없습니다.".format(img_fname), None
		car_no, xml = self.findCarnoInPicture(car_img, img_fname, easy)
		Cstool.log(2,"find carno[{}]".format(car_no))
		return car_no, xml

	def VocXml(file_name, width, height, plate_cont, plate_width, plate_height, char_rect_list, car_image=None):
		xmin = min([r[0] for r in plate_cont])
		ymin = min([r[1] for r in plate_cont])
		xmax = max([r[0] for r in plate_cont])
		ymax = max([r[1] for r in plate_cont])

		xml = []
		xml.append("<annotation>")
		xml.append("    <folder>carno</folder>")
		xml.append("    <filename>{}</filename>".format(file_name))
		xml.append("    <source>")
		xml.append("            <database>carno</database>")
		xml.append("            <annotation>carno</annotation>")
		xml.append("            <image>flickr</image>")
		xml.append("    </source>")
		xml.append("    <size>")
		xml.append("            <width>{}</width>".format(width))
		xml.append("            <height>{}</height>".format(height))
		xml.append("            <depth>3</depth>")
		xml.append("    </size>")
		xml.append("    <segmented>0</segmented>")
		xml.append("    <object>")
		xml.append("            <name>c</name>")
		xml.append("            <pose>Unspecified</pose>")
		xml.append("            <truncated>0</truncated>")
		xml.append("            <difficult>0</difficult>")
		xml.append("            <bndbox>")
		xml.append("                    <xmin>{}</xmin>".format(xmin))
		xml.append("                    <ymin>{}</ymin>".format(ymin))
		xml.append("                    <xmax>{}</xmax>".format(xmax))
		xml.append("                    <ymax>{}</ymax>".format(ymax))
		xml.append("            </bndbox>")
		if  not car_image is None:
			cv.rectangle(car_image, (xmin, ymin), (xmax, ymax), (255,255,0), 2)
		for r in char_rect_list:
			pxmin, pymin = Cstool.getOrgPostion(plate_cont, plate_width, plate_height, r[1].x,        r[1].y)
			pxmax, pymax = Cstool.getOrgPostion(plate_cont, plate_width, plate_height, r[1].x+r[1].w, r[1].y+r[1].h)
			
			xml.append("            <part>")
			xml.append("                    <name>{}</name>".format(r[0]))
			xml.append("                    <bndbox>")
			xml.append("                            <xmin>{}</xmin>".format(pxmin))
			xml.append("                            <ymin>{}</ymin>".format(pymin))
			xml.append("                            <xmax>{}</xmax>".format(pxmax))
			xml.append("                            <ymax>{}</ymax>".format(pymax))
			xml.append("                    </bndbox>")
			xml.append("            </part>")
			if not car_image is None:
				cv.rectangle(car_image, (pxmin, pymin), (pxmax, pymax), (0,0,255), 2)
		xml.append("    </object>")
		xml.append("</annotation>")
		if not car_image is None:
			Cstool.imshow(1, 'VocXml', car_image, 0, 0, 2000, is_wait=True)
		return '\n'.join(xml)

##################################################################################################################
def usage_exit():
	print("python reco.py train_id image_file [-g1 ~ -g99]   # 이미지 파일 분석하기 ")
	print("	      train_id : 학습정보 id")
	print("       image_file : 분석대상파일")
	print("       -g숫자 : 동작 과정 노출 여부")
	print("")
	sys.exit(0)

if __name__ == '__main__':
	argv = sys.argv
	if len(argv) < 2:
		usage_exit()
	train_id = argv[1]
	if not os.path.isfile("dnn/"+train_id+".txt") :
		print("dnn/{}.txt 파일이 없습니다".format(train_id))
		sys.exit(0)
	debug = 0
	if len(argv) >= 4 and argv[3].startswith("-g") and len(argv[3]) >= 2:
		debug  = int(argv[3][2:])
		Cstool.setDebugLevel(debug)
	image_file = argv[2]

	carno = Carno(train_id, debug)
	car_no, xml = carno.findProcess(image_file)
	if car_no != None :
		f = open(image_file.replace(".jpg", ".xml"), "w")
		f.write(xml)
		f.close()
		print('{}	{}'.format(image_file, car_no))
	else :
		car_no, xml = carno.findProcess(image_file, easy=True)
		if car_no != None :
			f = open(image_file.replace(".jpg", ".xml"), "w")
			f.write(xml)
			f.close()
			print('{}	{}  W'.format(image_file, car_no))
		else :
			print('{}	fail'.format(image_file))
	
