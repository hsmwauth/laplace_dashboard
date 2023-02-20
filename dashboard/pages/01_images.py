#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:25:28 2023
@author: auth

On this page it should be possible:
    * to toggle through all images (slider) and the already received images get displayd in a dummy-image.
    * display the corresponding table of "particlewise.feather"
"""
import helper_one as h
import constants as c
import streamlit as st
import pandas as pd

# write title
st.title('Images over time')

#write Information what this page is about:
st.info('Toggle through time to display all correspondent particles we received until now. The not received particles are represented through a place holder.')

# Getting database -> old one, before using the session state
# db = pd.read_feather(c.DBPATH)

# sort by filename
db = st.session_state['db'].sort_values(by = 'filename')

# get rid of duplicates
list_of_filenames = db['filename'].drop_duplicates()
#list_of_filenames = list_of_filenames

# get filenumber (just the 6digit number) from slider
frame = st.select_slider('', list_of_filenames)
fn = frame.split('Camera')[1]

# UPDATE session state FILENAME on click
changesessionstateframe = st.button('change sessionstate to this frame')
if changesessionstateframe:
    st.session_state['frame'] = frame

add2imagelist = st.button('add this frame to imagelist')
if add2imagelist:
    st.session_state['imagelist'].append(frame + '.png')

# get the dummy-image
[img, db_img] = h.fake_img(fn)

# display the dummy-image
st.header('Os7-S1 Camera' + str(fn) + '.png')
st.image(img)

#intitializing the database to display
db_display = db_img["order"]

# checkboxes for displaying the table
#interestingness
interest = st.sidebar.checkbox("interestingness", True)
if interest:
    db_display = pd.concat([db_display, db_img['interestingness']], axis=1)
    
# size_pixelcount
size = st.sidebar.checkbox("size_pixelcount", True)
if size:
    db_display = pd.concat([db_display, db_img['size_pixelcount']], axis=1)
    
# sharpness
sharpness = st.sidebar.checkbox("sharpness", True)
if sharpness:
    db_display = pd.concat([db_display, db_img['sharpness']], axis=1)
    
# boxposition and size
# box = st.sidebar.checkbox("Box position and dimensions", False)
# if box:
#    db_display = pd.concat([db_display, db_img['xpos'], db_img['ypos'], db_img['width'], db_img['height']], axis=1)

# write the table corresponding to the image
st.write(db_display)