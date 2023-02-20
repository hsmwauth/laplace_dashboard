import constants as c
import helper_one as h
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import cm
import os.path
from os.path import exists
from skimage import io
import matplotlib.pyplot as plt

#Getting DB
db = pd.read_feather(c.DBPATH)

# datapreparation for displaying the featurspace
[images_received, images_notreceived] = h.getfeaturespace(db)

# displaying properties
opacity = 0.1
size = 3

# creating the plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(images_notreceived[0],images_notreceived[1], c = 'k', alpha=opacity, s=size) #beachte reihenfolge, da die ersten übermahlt werden.
ax.scatter(images_received[0], images_received[1], c= 'g', alpha=opacity, s=size)
plt.legend(['not received images', 'received images'])
plt.xlabel('size')
plt.ylabel('sharpness')

#[x1, x2] = h.getinterestingness(db)

plt.show()