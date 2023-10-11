print(f'{"-"*20}开始加载第三方库{"-"*20}')
import os
import cv2
import time
from pathlib import Path
from easyocr import Reader
from PIL import Image, ImageDraw, ImageFont


class OcrReader(object):
    def __init__(self):
        self.lang_list = ['ch_sim']
        self.gpu = True
        self.model_storage_directory = './easyocr_weights'
        self.download_enabled = False
        self.detector = True
        self.recognizer = True
        self.reader = self.load_reader()

    def load_reader(self):
        reader = Reader(
            lang_list=self.lang_list,  # 语言
            gpu=self.gpu,  # GPU O/F
            model_storage_directory=self.model_storage_directory,  # 权重文件位置
            download_enabled=self.download_enabled,  # Download O/F
            detector=self.detector,  # 检测器 O/F
            recognizer=self.recognizer)  # 识别器 O/F
        return reader


class PicTextRecognizer(object):
    def __init__(self, reader, img_dir, img_name, res_dir, save=True, show=False):
        # 加载Reader
        print(f'{"-"*20}开始加载Reader{"-"*20}')
        self.reader = reader
        # 图片相关
        print(f'{"-" * 20}开始加载相关图片{"-" * 20}')
        self.img_dir = img_dir  # 图片所在文件夹
        self.img_name = img_name  # 图片名
        self.img_path = self.__load_img_path()  # 图片路径
        self.img = self.__load_img()  # 加载原图片
        self.pil_img = self.__load_pil_img()  # PIL格式图片
        self.pil_draw = self.__load_pil_draw()  # PIL画图对象
        self.h, self.w, self.processed_img = self.__load_processed_img()  # 处理后图片用于OCR识别

        # 画图参数
        self.color_hl = (225, 161, 5)  # 艳黄 RGB
        self.fontsize = 20  # 字体大小
        self.ttf_dir = 'ttf'  # 字体文件夹
        self.ttf_name = 'YaHei.ttf'  # 字体名字
        self.ttf_path = str(Path(self.ttf_dir) / Path(self.ttf_name))  # 字体路径
        self.font = ImageFont.truetype(self.ttf_path, size=self.fontsize)  # 定义字体

        # 加载结果
        print(f'{"-" * 20}正在生成识别结果{"-" * 20}')
        self.contents = self.__load_results()  # 加载结果
        self.res_dir = res_dir  # 结果保存文件夹
        self.res_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())  # 结果保存时间作为文件名
        self._res_dir = str(Path(self.res_dir) / Path(self.res_time))  # 结果保存子文件夹
        os.makedirs(self._res_dir, exist_ok=True)  # 创建结果保存文件夹
        self.res_name = f'{self.res_time}.jpg'  # 图片名
        self.res_path = str(Path(self._res_dir) / Path(self.res_name))  # 图片路径
        self.txt_name = f'{self.res_time}.txt'  # 文本名
        self.txt_path = str(Path(self._res_dir) / Path(self.txt_name))  # 文本路径
        self.s = '左上坐标|右下坐标|结果|置信度\n'  # 表头
        print(self.s, end='')
        self.draw_results()
        if save:
            self.save_results()
        if show:
            self.show_results()

    def __load_img_path(self):
        img_dir = self.img_dir  # 图片所在文件夹
        img_name = self.img_name  # 图片名
        return str(Path(img_dir) / Path(img_name))  # 图片路径

    def __load_img(self):
        return cv2.imread(str(self.img_path))  # 读取图片

    def __load_pil_img(self):
        return Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))  # 转为PIL图片

    def __load_pil_draw(self):
        return ImageDraw.Draw(self.pil_img)  # 创建PIL画图

    def __load_processed_img(self):
        h, w = self.img.shape[:2]  # 图片尺寸
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)  # 二值化
        return h, w, img

    def __load_results(self):
        return self.reader.readtext(self.processed_img)  # 文字识别

    def draw_results(self):
        color = self.color_hl  # 颜色
        fontsize = self.fontsize  # 字体大小
        draw = self.pil_draw  # 画图类
        font = self.font  # 字体
        for bbox, text, confidence in self.contents:  # 遍历结果
            x_min, y_min = bbox[0]  # 左下点坐标
            x_max, y_max = bbox[2]  # 右下点坐标
            _s = f'{bbox[0]}|{bbox[2]}|{text}|{confidence}\n'
            print(_s, end='')
            self.s += _s
            s = f"{text} ({confidence:.2f})"  # 所要添加的文本
            draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)  # 在图像上绘制框
            draw.text((x_min, y_min - (fontsize + 6)), s, fill=color, font=font)

    def show_results(self):
        self.pil_img.show()  # 展示结果

    def save_results(self):
        self.pil_img.save(self.res_path)  # 保存识别结果图片
        with open(self.txt_path, 'w', encoding='utf-8') as fp:
            fp.write(self.s)  # 写入识别结果文字


if __name__ == '__main__':
    rd = OcrReader().reader
    tr = PicTextRecognizer(reader=rd,
                        img_dir='images',
                        img_name='traffic_sign.jpg',
                        res_dir='0_res',
                        save=True,
                        show=False)
