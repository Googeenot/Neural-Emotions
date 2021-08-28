import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QMainWindow, QFileDialog, QStackedLayout, QAction, QMessageBox, QProgressBar, QScrollArea
from PyQt5.QtGui import QIcon, QPalette, QColor, QLinearGradient, QBrush, QFont, QDesktopServices, QPixmap, QImage, qRed, qBlue, qGreen, qRgb
from PyQt5.QtCore import QUrl, QTimer, QThread, QByteArray, QBuffer, QIODevice, pyqtSignal
from PyQt5.QtMultimedia import QCamera, QCameraInfo
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QCameraImageCapture, QAbstractVideoSurface, QVideoFrame, QAbstractVideoBuffer
import inspect
from tensorflow import keras
import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/device:CPU:0'):
    model = keras.models.load_model('model')


class VideoSurface(QAbstractVideoSurface):
    picture = pyqtSignal(QImage)

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def supportedPixelFormats(self, handle_type):
        result = []
        if handle_type == QAbstractVideoBuffer.NoHandle:
            result = [QVideoFrame.Format_RGB32, ]

        return result

    def present(self, frame: QVideoFrame):
        copy = QVideoFrame(frame)
        copy.map(QAbstractVideoBuffer.ReadOnly)

        image_format = QVideoFrame.imageFormatFromPixelFormat(copy.pixelFormat())

        my_image = QImage(int(copy.bits()), copy.width(), copy.height(), copy.bytesPerLine(), QImage.Format(image_format))

        copy.unmap()
        # Do something with your new `QImage` here!
        self.picture.emit(my_image.copy())
        return True


class AppHR(QMainWindow):

    def __init__(self):
        super().__init__()
        self.available_cameras = QCameraInfo.availableCameras()
        self.webcam = QCamera(self.available_cameras[0])
        self.video_widget = QLabel()
        self.viewport = VideoSurface(self.video_widget)
        self.viewport.picture.connect(self.paint_frames)

        self.checker = 0

        # self.viewport.setFixedSize(800, 500)
        # self.viewport.setFixedSize(200, 100)

        self.initial_widget = QWidget()
        self.initial_widget.setAutoFillBackground(True)
        palette = self.initial_widget.palette()
        gradient = QLinearGradient(-100, 1000, 1800, -100)
        gradient.setColorAt(0.0, QColor(0, 51, 255))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.initial_widget.setPalette(palette)
        self.page = QGridLayout(self.initial_widget)

        self.initui()

    def initui(self):
        self.resize(1200, 700)
        self.setWindowTitle('AI HR-agent')
        print(inspect.getmodule(QCameraImageCapture).__file__)

        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu('Help')

        self.first_screen()

    def changer(self):
        self.checker = 1

    def capture(self, img):
        rgb_pic = np.zeros((1, 128, 128, 3))
        gray_color_table = [qRgb(i, i, i) for i in range(256)]

        img_copy = img.copy()
        img_copy = img_copy.scaled(128, 128)

        for y in range(img_copy.height()):
            for x in range(img_copy.width()):
                rgb_pic[0][y][x][0] = qRed(img_copy.pixel(x, y))
                rgb_pic[0][y][x][1] = qGreen(img_copy.pixel(x, y))
                rgb_pic[0][y][x][2] = qBlue(img_copy.pixel(x, y))

        mask = np.flip(tf.round(model(rgb_pic)).numpy()[0, :, :, 0], 0)
        mask = np.require(mask, np.uint8, 'C') * 255
        mask_qt = QImage(mask.data, 128, 128, mask.strides[0], QImage.Format_Indexed8)
        mask_qt.setColorTable(gray_color_table)
        mask_qt = mask_qt.scaled(640, 480)

        img = img.mirrored()

        for y in range(mask_qt.height()):
            for x in range(mask_qt.width()):
                if qRed(mask_qt.pixel(x, y)) == 0 and qGreen(mask_qt.pixel(x, y)) == 0 and qBlue(mask_qt.pixel(x, y)) == 0:
                    img.setPixelColor(x, y, QColor(0, 0, 0))

        test_frame = QLabel()
        test_frame.setPixmap(QPixmap.fromImage(img))
        self.page.addWidget(test_frame, 1, 0)

    def paint_frames(self, img):
        if self.checker == 1:
            self.capture(img.copy())
            self.checker = 0
        self.video_widget.setPixmap(QPixmap.fromImage(img.mirrored()))
        self.video_widget.repaint()

    def first_screen(self):
        temp_label = QPushButton('Capture')
        temp_label.setFont(QFont("San-Serif", 17))
        temp_label.setStyleSheet('''border: 2px solid #ffffff; border-radius: 10px; background: rgba(255, 255, 255, 0.8); color: rgb(0, 150, 115); padding-left: 30px; padding-right: 30px; padding-top: 10px; padding-bottom: 10px;''')
        temp_label.clicked.connect(self.changer)

        self.page.addWidget(self.video_widget, 0, 0)
        self.page.addWidget(temp_label, 0, 1)
        self.setCentralWidget(self.initial_widget)

        self.webcam.unload()
        self.webcam.setViewfinder(self.viewport)
        self.webcam.setCaptureMode(QCamera.CaptureStillImage)
        self.webcam.start()

        self.show()















if __name__ == '__main__':

    app = QApplication([])
    application = AppHR()
    sys.exit(app.exec_())
