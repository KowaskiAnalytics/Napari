import napari
from skimage.io import imread
from qtpy.QtWidgets import QMainWindow
from qtpy import uic
from pathlib import Path
from aicsimageio import AICSImage

# Define the main window class
class FancyGUI(QMainWindow):
    def __init__(self, napari_viewer):          # include napari_viewer as argument (it has to have this name)
        super().__init__()
        self.viewer = napari_viewer
        self.UI_FILE = str(Path(__file__).parent / "UI.ui")  # path to .ui file
        uic.loadUi(self.UI_FILE, self)           # load QtDesigner .ui file
        print(self.viewer.layers)

        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))

    def updatelayer(self):
        self.comboBox_layer.clear()
        for i in self.viewer.layers:
            self.comboBox_layer.addItem(str(i))


viewer = napari.Viewer()
napari_image = imread('Artboard4.png')      # Reads an image from file
viewer.add_image(napari_image, name='napari_island')                # Adds the image to the viewer and give the image layer a name

flood_widget = FancyGUI(viewer)                                     # Create instance from our class
viewer.window.add_dock_widget(flood_widget, area='right')           # Add our gui instance to napari viewer


@viewer.layers.events.removed.connect
@viewer.layers.events.inserted.connect
@viewer.layers.events.changed.connect
def updatelayer(event):
    flood_widget.updatelayer()


napari.run()