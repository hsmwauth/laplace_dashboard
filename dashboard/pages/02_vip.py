#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 20 09:35:11 2023
@author: auth

Was soll hier gezeigt werden:
    * Hier sollen die Interessantesten gefundenen Partikel gezeigt werden
    * evtl. hier durch Order mit slider togglen
"""

import streamlit as st
import pandas as pd

import helper_one as h
import constants as c

# Set title and Info for this slide
st.title('Images over order')
st.info('Displaying the Images due to their order (not due to timestamp).')

# importing database and sorting
# db = pd.read_feather(c.DBPATH)
db = st.session_state['db'].sort_values(by='order')

# Create Slider
order = st.select_slider('Pic an order integer',db['order'] )

# from order to filename
db =st.session_state['db'][st.session_state['db']['order'] == order]
filename = db['filename'].tolist()[0] # get filename from pandas series

# UPDATE SESSION STATE to this order
changesessionstateorder = st.button('change the session state to this order')
if changesessionstateorder:
    st.session_state['order'] = order

# Add this file to filename
add2imagelist = st.button('add this frame to imagelist')
if add2imagelist:
    st.session_state['imagelist'].append(filename + '.png')


st.write(filename)

# extract the imagenumber as str
fn = filename.split('Camera')[1]

# get the dummy-image
[img, db_img] = h.fake_img(fn)

# displaying the image
st.image(img)



# displaying the correspondent database with all particles in this image
st.write(db_img)
