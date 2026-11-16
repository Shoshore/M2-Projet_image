from src.ai_based_method import stegano_gan
from src.evaluation import utils
from src.interface import interface
from PIL import Image
from os import listdir
from os.path import splitext, join
tools = utils.Utils()
D = 2
FILE = "./assets/images/"
a = stegano_gan.Stegano_gan(D = D, not_train = True)

from tkinter import * 
from tkinter.filedialog import *

window = interface.mainWindow(a.hide_adaptive)
window.start()
