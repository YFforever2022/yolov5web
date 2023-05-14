# This code is used for WEB requests to return inference results
import base64
import threading
import time
import torch
import numpy as np
import os
import sys
from models.common import DetectMultiBackend
from utils.general import (Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from gevent.pywsgi import WSGIServer
from threading import Thread
from flask import Flask, request
import urllib.request
import mmap
import ctypes
import ctypes.wintypes
import win32api
import win32con
import win32gui
import win32ui

lock = threading.Lock()
jcqbh = "1"
weights = 'yolov5n.pt'
IMGSZ = [640, 640]
max_det = 1000
conf_thres = 0.1
iou_thres = 0.45
device_ = ''
dk = 7700
temp_device = ''

geshu = len(sys.argv)
if geshu < 6:
    print('参数不足，启动失败，2秒后自动退出')
    time.sleep(2)
    sys.exit()
else:
     jcqbh = sys.argv[1]
     SIZE = sys.argv[2]
     IMGSZ = [int(SIZE), int(SIZE)]
     dk = int(sys.argv[3])
     temp_device = sys.argv[4]
     if temp_device == 'aidmlm':
         device_ = ''
     else:
         device_ = temp_device
     weights = sys.argv[5]

pid = os.getpid()

print('jcqbh', jcqbh)
print('weights', weights)
print('IMGSZ', IMGSZ)

def Window_Shot(hwnd):
    if win32gui.IsWindow(hwnd) == False:
        hwnd = win32gui.GetDesktopWindow()
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        w = MoniterDev[0][2][2]
        h = MoniterDev[0][2][3]
    else:
        ret = win32gui.GetClientRect(hwnd)
        w = ret[2]
        h = ret[3]

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

    signedIntsArray = saveBitMap.GetBitmapBits(True)
    im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
    im_opencv.shape = (h, w, 4)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return im_opencv


def get_model():
    start_time = time.time()
    device1 = select_device(device_)
    model1 = DetectMultiBackend(weights, device=device1)
    stride1, names1, pt1 = model1.stride, model1.names, model1.pt

    end_time = time.time()
    print("模型加载完成,耗时", int((end_time-start_time)*1000), "毫秒")
    print('')

    return device1, model1, stride1, names1, pt1

device, model, stride, names, pt = get_model()
imgsz = check_img_size(IMGSZ, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

@torch.no_grad()
def yuce(img0):
    if hasattr(img0, 'shape') == False:
        return ''

    lock.acquire()
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    im0 = img0
    if im0 is None:
        return ''

    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    det = pred[0]

    xywh_list = ''
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            w = abs(x2-x1)
            h = abs(y2-y1)
            zxd = float(conf)
            xywh = str(c) + "," + str(x1) + "," + str(y1) + "," + str(w) + "," + str(h) + "," + str(zxd) + ',' + str(names[c]) + '|'
            xywh_list = xywh_list + xywh
    xywh_list = xywh_list.encode('utf-8')
    lock.release()
    return xywh_list


def bytesToMat(img):
    np_arr = np.frombuffer(bytearray(img), dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

app = Flask(__name__)


@app.route('/pid', methods=['GET', 'POST'])
def YOLOv5_WEB_pid():
    return str(pid)


@app.route('/pic', methods=['GET', 'POST'])
def YOLOv5_WEB_pic():
    if request.data == b'':
        return ''
    try:
        tp = request.get_data()
        img = bytesToMat(tp)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''

@app.route('/hwnd',methods=['GET', 'POST'])
def YOLOv5_WEB_hwnd():
    try:
        tp = request.get_data()
        sj = tp.decode('utf-8', "ignore")
        jb = int(float(sj))
        img = Window_Shot(jb)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''

@app.route('/base64', methods=['GET', 'POST'])
def YOLOv5_WEB_base64():
    try:
        tp = request.get_data()
        tp_wb = tp.decode('utf-8', "ignore")
        b_tp = base64.b64decode(tp_wb)
        img = bytesToMat(b_tp)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''

@app.route('/file',methods=['GET', 'POST'])
def YOLOv5_WEB_file():
    try:
        tp = request.get_data()
        tp_file = tp.decode('utf-8', "ignore")
        img = cv2.imdecode(np.fromfile(file=tp_file, dtype=np.uint8), -1)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''



def qidong():
    print('使用端口号' + str(dk) + '启动WEB服务器')
    print('')
    WSGIServer(('0.0.0.0', dk), app, log=None).serve_forever()  # log=None



SendMessage = ctypes.windll.user32.SendMessageA

class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_void_p)
    ]

PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)



class Listener:
    def __init__(self):
        WindowName = "aidmlm.com" + jcqbh
        message_map = {
            win32con.WM_COPYDATA: self.OnCopyData
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = WindowName
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            "aidmlm.com" + jcqbh,
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        print('pid', pid)
        print("hwnd", self.hwnd)
        Thread(target=qidong).start()


    def OnCopyData(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
        s = ctypes.string_at(pCDS.contents.lpData, pCDS.contents.cbData).decode() # "utf-8", "ignore"
        cd = int(float(s))

        if wparam == 1:

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, cd, file_name, mmap.ACCESS_WRITE)  # python读取共享内存
            tp = shmem.read(cd)  # 从共享内存读取共享数据

            img = bytesToMat(tp)
            jieguo = yuce(img)

            # 将jieguo写到共享内存

            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)

            shmem.close()  # 关闭映射

            return jgcd
        if wparam == 0:
            return pid
        if wparam == 2:
            jb = cd
            if win32gui.IsWindow(jb) == False:
                return 11
            # img = screen.grabWindow(jb).toImage()
            # img = convertQImageToMat(img)  # QImage转Mat格式
            img = Window_Shot(jb)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            jieguo = yuce(img)

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, 51200, file_name, mmap.ACCESS_WRITE)  # python读取共享内存
            # 将jieguo写到共享内存

            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)
            shmem.close()  # 关闭映射

            return jgcd
        return 10


l = Listener()
win32gui.PumpMessages()
