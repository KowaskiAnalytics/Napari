"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QWidget, QFileDialog
from qtpy.QtGui import QPixmap
import qtpy.QtCore
from qtpy import uic
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import pandas as pd
import cv2
from napari.qt.threading import WorkerBase, WorkerBaseSignals

if TYPE_CHECKING:
    import napari

from napari_plugin_engine import napari_hook_implementation
from pathlib import Path

from functions import *
from autocorrelation import cycledegreesAuto
from crosscorrelation import cycledegreesCross
import ClickLabel

class ArrayShapeIncompatible(Exception):
    """Raised when the input value is too small"""
    pass


# Define the main widget window for the plugin
class AutocorrelationTool(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.UI_FILE = abspath(__file__, 'static/form.ui')  # path to .ui file
        uic.loadUi(self.UI_FILE, self)

        # Updates comboboxes when a layer is added, removed or updated
        self.updatelayer()
        self.viewer.layers.events.removed.connect(self.updatelayer)
        self.viewer.layers.events.inserted.connect(self.updatelayer)
        self.viewer.layers.events.changed.connect(self.updatelayer)

        self.thread = None
        self.inputarray = None
        self.maskedarray = None
        self.maskedarray_c = None
        self.outputPath = None
        self.threshmask = None
        self.restrictdeg = [-90, 90]
        self.doCross = False

        self.label_19.setVisible(False)
        self.comboBox_layer_1.setVisible(False)
        self.radioButton_1.setVisible(False)
        self.radioButton_2.setVisible(False)
        self.radioButton_1.toggled.connect(self.corToggle)

        self.label_A.clicked.connect(self.corImgA)
        self.label_C.clicked.connect(self.corImgC)
        self.genThresh.clicked.connect(self.visualizeThresh)

        self.pushButton_genShapes.clicked.connect(self.createshapelayer)

        self.comboBox_mode.currentIndexChanged.connect(self.changeLock_zone)
        self.comboBox_gridsplit.currentIndexChanged.connect(self.changeLock_grid)
        self.comboBox_visOutput.currentIndexChanged.connect(self.changeLock_vis)
        self.angleSlider.valueChanged.connect(self.updateslidervalue)
        self.analyze.clicked.connect(self.Autocorrelate)
        self.pushButton_File.clicked.connect(self.filedialog)

    def corImgA(self):
        self.radioButton_1.click()
        self.label_A.setPixmap(QPixmap("static/auto_T.png"))
        self.label_C.setPixmap(QPixmap("static/cross_F.png"))


    def corImgC(self):
        self.radioButton_2.click()
        self.label_A.setPixmap(QPixmap("static/auto_F.png"))
        self.label_C.setPixmap(QPixmap("static/cross_T.png"))

    # Toggle between autocorrelation and cross-correlation
    def corToggle(self):
        if self.radioButton_1.isChecked():
            self.label_19.setVisible(False)
            self.comboBox_layer_1.setVisible(False)
            self.doCross = False

        else:
            self.label_19.setVisible(True)
            self.comboBox_layer_1.setVisible(True)
            self.doCross = True

    def updateslidervalue(self):
        self.sliderLabel.setText(str(self.angleSlider.value()))

    # File dialogue for output images and csv
    def filedialog(self):
        self.outputPath = QFileDialog.getExistingDirectory(self, 'Select output path')

    # Locks restricted angle parameters when full search is enabled
    def changeLock_zone(self):
        if self.comboBox_mode.currentText() == "Full search":
            self.spinBox_zoneMid.setEnabled(False)
            self.angleSlider.setEnabled(False)
            self.restrictdeg = [-90, 90]

        else:
            self.spinBox_zoneMid.setEnabled(True)
            self.angleSlider.setEnabled(True)
            self.restrictdeg = [self.spinBox_zoneMid.value() - self.angleSlider.value(),
                                self.spinBox_zoneMid.value() + self.angleSlider.value()]

    # Locks grid split option when None is selected
    def changeLock_grid(self):
        if self.comboBox_gridsplit.currentText() == "None":
            self.spinBox_gridLeft.setEnabled(False)
            self.spinBox_gridRight.setEnabled(False)

        else:
            self.spinBox_gridLeft.setEnabled(True)
            self.spinBox_gridRight.setEnabled(True)

    # Locks visualization paraneters when None is selected
    def changeLock_vis(self):
        if self.comboBox_visOutput.currentText() == "None":
            self.doubleSpinBox_visLeft.setEnabled(False)
            self.doubleSpinBox_visRight.setEnabled(False)

        else:
            self.doubleSpinBox_visLeft.setEnabled(True)
            self.doubleSpinBox_visRight.setEnabled(True)

    # Updates comboboxes
    def updatelayer(self):
        self.comboBox_layer.clear()
        self.comboBox_layer_1.clear()

        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))
            self.comboBox_layer_1.addItem(str(i))

    def threshold(self):
        def do(input):
            blurredz = np.array(input, dtype='uint8')
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

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            maskedarray = copy.deepcopy(input)

            ### overlay original image with mask
            maskedarray[mask == 0] = np.nan

            return maskedarray, mask

        self.maskedarray, self.threshmask = do(self.inputarray)
        if self.doCross:
            self.maskedarray_c = copy.deepcopy(self.inputarray_c)
            self.maskedarray_c[self.threshmask == 0] = np.nan

    def readfile(self):
        ### Read selected layer
        inputarray = self.viewer.layers[self.comboBox_layer.currentText()].data
        ### Set to greyscale if needed
        try:
            inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass

        self.inputarray = np.array(inputarray, dtype='float32')

        if self.doCross:
            inputarray = self.viewer.layers[self.comboBox_layer_1.currentText()].data
            ### Set to greyscale if needed
            try:
                inputarray = cv2.cvtColor(inputarray, cv2.COLOR_RGB2GRAY)
            except Exception:
                pass

            self.inputarray_c = np.array(inputarray, dtype='float32')

    def visualizeThresh(self):
        self.readfile()
        self.threshold()
        mask = self.threshmask
        self.viewer.add_image(mask,
                              name=str(self.viewer.layers[self.comboBox_layer.currentText()]) + "_mask",
                              colormap="cyan",
                              opacity=0.30)

    def createshapelayer(self):
        self.viewer.add_shapes(shape_type="rectangle", edge_width=5, edge_color='#05d9b9', face_color='#05e69f',
                               opacity=0.4, name=str(self.viewer.layers[self.comboBox_layer.currentText()]) + "_ROI")

    def updateprogress(self, progress):
        if progress[0]:
            self.progressBar.setValue(100)
        else:
            self.progressBar.setValue(self.progressBar.value() + (1 / int(progress[1])) * 100)

    def Autocorrelate(self):
        # print(self.viewer.layers[self.comboBox_layer.currentText() + "_ROI"].data)
        self.readfile()
        if self.doCross and np.shape(self.inputarray) != np.shape(self.inputarray_c):
            raise ArrayShapeIncompatible("Selected layers should have the same shape")
        else:
            self.threshold()
            self.changeLock_zone()
            self.thread = MyWorker()
            self.thread.updateparameters(currentlayer=self.comboBox_layer.currentText(),
                                         maskedarray=self.maskedarray,
                                         maskedarray_c=(self.maskedarray_c if self.doCross else None),
                                         mode=self.comboBox_mode.currentText(),
                                         gridsplitmode=self.comboBox_gridsplit.currentText(),
                                         gridsplitleft=self.spinBox_gridLeft.value(),
                                         gridsplitright=self.spinBox_gridRight.value(),
                                         autocormode=self.comboBox_AutocorMethod.currentText(),
                                         visoutput=self.comboBox_visOutput.currentText(),
                                         visleft=self.doubleSpinBox_visLeft.value(),
                                         visright=self.doubleSpinBox_visRight.value(),
                                         pixelsize=self.spinBox_pixel.value(),
                                         outimg=self.checkBox_outImg.isChecked(),
                                         outcsv=self.checkBox_outCSV.isChecked(),
                                         path=self.outputPath,
                                         restrictdeg=self.restrictdeg)

            self.progressBar.setValue(0)
            self.thread.work()

            self.thread.progress.connect(self.updateprogress)


class MyWorkerSignals(WorkerBaseSignals):
    progress = qtpy.QtCore.Signal(object)


class MyWorker():
    # progress = qtpy.QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self.maskedarray_c = None
        self.currentlayer = None
        self.path = None
        self.outcsv = None
        self.outimg = None
        self.pixelsize = None
        self.visright = None
        self.visleft = None
        self.visoutput = None
        self.autocormode = None
        self.gridsplitright = None
        self.gridsplitleft = None
        self.gridsplitmode = None
        self.analysismode = None
        self.maskedarray = None
        self.restrictdeg = None

    def updateparameters(self,
                         currentlayer,
                         maskedarray,
                         maskedarray_c,
                         mode,
                         gridsplitmode,
                         gridsplitleft,
                         gridsplitright,
                         autocormode,
                         visoutput,
                         visleft,
                         visright,
                         pixelsize,
                         outimg,
                         outcsv,
                         path,
                         restrictdeg):

        self.currentlayer = currentlayer
        self.maskedarray = maskedarray
        self.maskedarray_c = maskedarray_c
        self.analysismode = mode
        self.gridsplitmode = gridsplitmode
        self.gridsplitleft = gridsplitleft
        self.gridsplitright = gridsplitright
        self.autocormode = autocormode
        self.visoutput = visoutput
        self.visleft = visleft
        self.visright = visright
        self.pixelsize = pixelsize
        self.outimg = outimg
        self.outcsv = outcsv
        self.path = path
        self.restrictdeg = restrictdeg

    def work(self):
        print("ok")
        gridsplitmode = self.gridsplitmode
        # gridsplitmode = "Auto"

        gridsplitval = [self.gridsplitleft, self.gridsplitright]
        # gridsplitval = [200,200]

        cleangrids = []

        if self.maskedarray_c is None:
            grids = gridsplit(self.maskedarray, gridsplitmode, gridsplitval)
            for index, grid in enumerate(grids):
                if not np.isnan(grid).all():
                    cleangrids.append(grid)

            indexgrids = []

            for index, grid in enumerate(cleangrids):
                indexgrids.append([index, grid])

            with Pool(4) as self.pool:
                output = []
                for _ in self.pool.imap_unordered(partial(cycledegreesAuto,
                                                          pxpermicron=self.pixelsize,
                                                          filename=self.currentlayer,
                                                          mode=self.autocormode,
                                                          outputimg=self.outimg,
                                                          outputcsv=self.outcsv,
                                                          restrictdeg=self.restrictdeg,
                                                          outputpath=self.path), indexgrids):
                    output.append(_)
                    # self.progress.emit([False, len(indexgrids)])
                    # print(self.progressBar.value())
                    # self.progressBar.setValue(self.progressBar.value() + 1)

                self.pool.close()
                self.pool.join()

                # self.progress.emit([True, len(indexgrids)])
                output = np.array(output)
                weighted_avg = np.average(output[:, 0], weights=output[:, 1])
                intervallist = output[:, 2]
                medianfrequency = np.average(output[:, 2], weights=output[:, 1])
                print('FINAL RESULT', weighted_avg)
                print(intervallist)
                print('most likely periodicity interval', medianfrequency)

                if not self.visoutput == "None":
                    df = pd.concat(output[:, 3])
                    PrincipleComponents(df, self.visoutput,
                                        (self.visleft, self.visright))
                if not self.outcsv == "None":
                    df = pd.concat(output[:, 3])
                    df2 = pd.DataFrame({"total grids": [np.shape(indexgrids)[0]]})
                    new = pd.concat([df, df2], axis=1)
                    new.to_csv(self.path + "/" + self.currentlayer + ".csv", sep=";")

        else:
            grids_a = gridsplit(self.maskedarray, gridsplitmode, gridsplitval)
            grids_c = gridsplit(self.maskedarray_c, gridsplitmode, gridsplitval)
            for index, grid in enumerate(grids_a):
                if not np.isnan(grids_a).all():
                    cleangrids.append([grids_a[index],grids_c[index]])

            indexgrids = []

            for index, grid in enumerate(cleangrids):
                indexgrids.append([index, grid])

            with Pool(4) as self.pool:
                output = []
                for _ in self.pool.imap_unordered(partial(cycledegreesCross,
                                                          pxpermicron=self.pixelsize,
                                                          filename=self.currentlayer,
                                                          mode=self.autocormode,
                                                          outputimg=self.outimg,
                                                          outputcsv=self.outcsv,
                                                          restrictdeg=self.restrictdeg,
                                                          outputpath=self.path), indexgrids):
                    output.append(_)
                    # self.progress.emit([False, len(indexgrids)])
                    # print(self.progressBar.value())
                    # self.progressBar.setValue(self.progressBar.value() + 1)

                self.pool.close()
                self.pool.join()

                # self.progress.emit([True, len(indexgrids)])
                output = np.array(output)

                if not self.visoutput == "None":
                    df = pd.concat(output[:, 0])
                    PrincipleComponents(df, self.visoutput,
                                        (self.visleft, self.visright))
                if not self.outcsv == "None":
                    print(output[0])
                    try:
                        df = pd.concat(output[:, 0])
                    except TypeError:
                        df = output[0]

                    df2 = pd.DataFrame({"total grids": [np.shape(indexgrids)[0]]})
                    new = pd.concat([df, df2], axis=1)
                    new.to_csv(self.path + "/" + self.currentlayer + ".csv", sep=";")

    def stop(self):
        self.terminate()
        self.pool.stop()


import napari


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return AutocorrelationTool


if __name__ == '__main__':
    viewer = napari.Viewer()

    Autocorrelation_widget = AutocorrelationTool(viewer)  # Create instance from our class
    viewer.window.add_dock_widget(Autocorrelation_widget, area='right')  # Add our gui instance to napari viewer


    @viewer.layers.events.removed.connect
    @viewer.layers.events.inserted.connect
    @viewer.layers.events.changed.connect
    def updatelayer(event):
        Autocorrelation_widget.updatelayer()


    napari.run()
