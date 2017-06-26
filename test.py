# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

img = Image.open("D:/Code/data/Invasive Species Monitoring/train/1.jpg")
img1 = img.resize((224, 224))
img.show()
img1.show()



a = 1