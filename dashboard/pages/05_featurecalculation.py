#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 08:42:17 2023

@author: auth

Hier werden die Features analysiert.
"""

import streamlit as st
import pandas as pd
import os
from skimage import io
import numpy as np
import cv2
from PIL import Image


import constants as c

st.title('Featureanalyse')

# importing database and sorting
db = pd.read_feather(c.DBPATH)
db = db.sort_values(by='order')

# Create Slider
order = st.select_slider('Pic an order integer',db['order'] )

# getting particle where order 
db = db[db['order'] == order]
fn = db['filename'].tolist()[0]


# getting Particle-crop
cropped_particle_filename = c.CROPPEDIMAGEPATH + '/' + str(order) + '_' + fn + '.png'

if os.path.exists(cropped_particle_filename):
    cropped_img = io.imread(cropped_particle_filename)
    
    # normalizing image
    cropped_image = np.asarray(cropped_img)/255 # normalizing dtype=uint8 to 1

    # CALCULATE FEATURE -> sharpness
    blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)  # apply Gauss-Filter
    blur_norm = (blur-np.min(blur))/(np.max(blur)-np.min(blur)) #normalize beween 0,1
    blur_uint8 = (blur_norm*255).astype(np.uint8)
    blur_im = Image.fromarray(blur_uint8)
    
    # LAPLACIAN
    laplacian = cv2.Laplacian(blur_norm, cv2.CV_64F)
    laplacian_norm = (laplacian-np.min(laplacian))/np.max(laplacian)-np.min(laplacian) # normalize between 0,1
    laplacian_log = np.log10(laplacian_norm)
    laplacian_lognorm = (laplacian_log-np.min(laplacian_log))/np.max(laplacian_log)-np.min(laplacian_log) # normalize between 0,1
    laplacian_uint8 = (laplacian_norm*255).astype(np.uint8)
    laplacian_im = Image.fromarray(laplacian_uint8)

    
#    blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)  # apply Gauss-Filter
#    laplacian = cv2.Laplacian(blur, cv2.CV_64F) # apply Laplace-Filter
#    sharpness_img = laplacian
#    st.write(sharpness_img.min())
#    sharpness_scalar = sharpness = laplacian.var()
    #sharpness = np.log10(sharpness) # TODO aply divie by zero and apply the log10
    

    # CALCULATE FEATURE -> size_pixelcount
    binary_image = np.where(cropped_image > c.threshold_constant, 1, 0)
    # n_ones= np.count_nonzero((binary_image) == 1) # count, wher no particle is ...
    n_zeros = np.count_nonzero(binary_image == 0) # ... or count the pixels where a particle is
    binary_image = (binary_image*255).astype(np.uint8)
    binary_image = Image.fromarray(binary_image)
    #size_scalar = n_zeros
    

    # display particle, sharpness_image, size_image
    st.image(cropped_image)
    st.image(blur_im)
    st.image(laplacian_im)
    st.image(binary_image)

    st.dataframe(db)

    
    
else:
    warningmessage = 'Image ' + cropped_particle_filename.split('/')[1] + 'not found'
    st.warning(warningmessage)
    


    