import cv2
import numpy as np

class cchecker():

    def initialize(self, img):
        '''Initialize the color checker area'''
        arr = np.fromstring(img, np.uint8)
        self.image = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
        self.detector = cv2.mcc.CCheckerDetector_create()
        self.detector.process(self.image, cv2.mcc.MCC24, 1)
        self.checkers = self.detector.getListColorChecker()
        self.initialized = (len(self.checkers) != 0)
        return (len(self.checkers) != 0)

    def run(self, checker_id=0):
        if self.initialized:
            checker = self.checkers[checker_id]
            cdraw = cv2.mcc.CCheckerDraw_create(checker)
            self.image_draw = self.image.copy()
            cdraw.draw(self.image_draw)
            
            chartsRGB = checker.getChartsRGB()
            width, height = chartsRGB.shape[:2]
            self.roi = chartsRGB[0:width,1]
            # print (roi)
            rows = int(self.roi.shape[:1][0])
            src = chartsRGB[:,1].copy().reshape(int(rows/3), 1, 3)
            src /= 255
            #print(src.shape)

            self.model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
            self.model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
            self.model.setCCM_TYPE(cv2.ccm.CCM_3x3)
            self.model.setDistance(cv2.ccm.DISTANCE_CIE2000)
            self.model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
            self.model.setLinearGamma(2.2)
            self.model.setLinearDegree(3)
            self.model.setSaturatedThreshold(0, 0.98)
            self.model.run()

    def infer(self, image=None):
        if image is not None:
            img = image
        else:
            img = self.image

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64)
        img = img/255
        calibratedImage = self.model.infer(img)
        out = calibratedImage * 255
        out[out < 0] = 0
        out[out > 255] = 255
        out = out.astype(np.uint8)
        self.image_out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def get_roi(self):
        return self.roi
    
    def get_output(self):
        return cv2.cvtColor(self.image_out, cv2.COLOR_BGR2RGB)
    
    def get_draw(self):
        return cv2.cvtColor(self.image_draw, cv2.COLOR_BGR2RGB)
    
    def get_init(self):
        return self.initialized

    def __init__(self, img):
        self.initialized = False
        if img is not None:
            self.initialize(img)

    def get_draw_plot_srgb(self):
        standard_color = [
            [115, 82, 68],
            [194, 150, 130],
            [98, 122, 157],
            [87, 108, 67],
            [133, 128, 177],
            [103, 189, 170],
            [214, 126, 44],
            [80, 91, 166],
            [193, 90, 99],
            [94, 60, 108],
            [157, 188, 64],
            [224, 163, 46],
            [56, 61, 150],
            [70, 148, 73],
            [175, 54, 60],
            [231, 199, 31],
            [187, 86, 149],
            [8, 133, 161]]

        c = 0
        rx = []
        ry = []
        gx = []
        gy = []
        bx = []
        by = []
        for i in standard_color:
            rx.append(i[0])
            gx.append(i[1])
            bx.append(i[2])
            ry.append(self.roi[c * 3 + 0])
            gy.append(self.roi[c * 3 + 1])
            by.append(self.roi[c * 3 + 2])
            c += 1
        return [rx, ry, gx, gy, bx, by]