import napari
from skimage.io import imread
from qtpy.QtWidgets import QMainWindow
from qtpy import uic
from pathlib import Path
import cv2
import numpy as np
import copy
import aicsimageio



# Define the main window class
class AutocorrelationTool(QMainWindow):
    def __init__(self, napari_viewer):          # include napari_viewer as argument (it has to have this name)
        super().__init__()
        self.viewer = napari_viewer
        self.UI_FILE = str(Path(__file__).parent / "UI.ui")  # path to .ui file
        uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file

        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))

        self.genThresh.clicked.connect(self.visualizeThresh)

        self.comboBox_mode.currentIndexChanged.connect(self.changeLock_zone)
        self.comboBox_gridsplit.currentIndexChanged.connect(self.changeLock_grid)



    def changeLock_zone(self):
        if self.comboBox_mode.currentText() == "Full search":
            self.spinBox_zoneLeft.setEnabled(False)
            self.spinBox_zoneRight.setEnabled(False)

        else:
            self.spinBox_zoneLeft.setEnabled(True)
            self.spinBox_zoneRight.setEnabled(True)

    def changeLock_grid(self):
        if self.comboBox_gridsplit.currentText() == "None":
            self.spinBox_gridLeft.setEnabled(False)
            self.spinBox_gridRight.setEnabled(False)

        else:
            self.spinBox_gridLeft.setEnabled(True)
            self.spinBox_gridRight.setEnabled(True)

    def updatelayer(self):
        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))

    def threshold(self):
        ### Set type, blur and threshold
        blurredz = np.array(self.inputarray, dtype='uint8')
        blurredz = cv2.GaussianBlur(blurredz, (5, 5), 5)

        ### MANUAL THRESHOLDING ###
        thresh = self.threshSlider.value()
        thresholdH = blurredz[:, :] > thresh
        thresholdL = blurredz[:, :] <= thresh
        blurredz[thresholdH] = 1
        blurredz[thresholdL] = 0

        edged = cv2.Canny(blurredz, 0, 1)

        contours, hierarchy = cv2.findContours(blurredz, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(blurredz.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        self.maskedarray = copy.deepcopy(self.inputarray)

        ### overlay original image with mask
        self.maskedarray[mask == 0] = np.nan

        return mask

    def readfile(self):
        ### Read selected layer
        inputarray = self.viewer.layers[self.comboBox_layer.currentText()].data
        print(type(inputarray))
        ### Set to greyscale if needed
        try:
            inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass

        self.inputarray = np.array(inputarray, dtype='float32')

    def visualizeThresh(self):
        self.readfile()
        mask = self.threshold()
        self.viewer.add_image(mask,
                              name = str(self.viewer.layers[self.comboBox_layer.currentText()]) + "_mask",
                              colormap = "cyan",
                              opacity = 0.30)

    def Autocorrelate(self):




if __name__ == '__main__':
    viewer = napari.Viewer()
    napari_image = imread('IMG0194_Kv1 BIC 48h.obf - STAR RED.tif')      # Reads an image from file
    viewer.add_image(napari_image, name='napari_island')                # Adds the image to the viewer and give the image layer a name

    Autocorrelation_widget = AutocorrelationTool(viewer)                                     # Create instance from our class
    viewer.window.add_dock_widget(Autocorrelation_widget, area='right')           # Add our gui instance to napari viewer


    @viewer.layers.events.removed.connect
    @viewer.layers.events.inserted.connect
    @viewer.layers.events.changed.connect
    def updatelayer(event):
        Autocorrelation_widget.updatelayer()


    napari.run()