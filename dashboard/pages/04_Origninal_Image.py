#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:35:44 2023

@author: auth

Hier soll der Vergleich zwischen dem dummy-image zusammengesetzt aus den gekropten partikeln und dem originalimage erfolgen.
Dies dient zur Kontrolle und kann bei der eigentlichen Anwendung nicht eingesetzt werden, da wir dann das Originalimage nicht haben.
"""

import streamlit as st
import pandas as pd
# import numpy as np
from skimage import io
# import os.path
# from PIL import Image

import constants as c
import helper_one as h


# sort by filename
db = st.session_state['db'].sort_values(by = 'filename')

# get rid of duplicates
list_of_filenames = db['filename']
list_of_filenames = list_of_filenames.drop_duplicates()

# Title and information
st.title('Original vs. Assembled Image')
st.info('Comparing the original image with the reassambled one to spot mistakes.')

# get filenumber (just the 6digit number) from slider
filename = st.select_slider('', list_of_filenames)
fn = filename.split('Camera')[1]

# get the dummy-image
[dummy_img, db_img] = h.fake_img(fn)

#Get the original image
path2orig = c.IMAGEPATH + '/' + filename + '.png'
orig_img = io.imread(path2orig)

title = filename + '.png'
st.header(title)
col1, col2 = st.columns(2)

# Displaying the original and the dummy image side by side
col1.caption('Original Image')
col1.image(orig_img)

col2.caption('Reassembled Image')
col2.image(dummy_img)

