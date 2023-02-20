#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:49:19 2023
@author: auth

Following questions should be answered on this page:
    * How many cropped images are received.
    * How many cropped images are we expecting to receive until certain time.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

import helper_one as h
import constants as c

# Set title and Info for this slide
st.title('Featurespace')
st.info('Displaying the featurespace on base what the AID detected. Received particles get displayed in green. Not received one in black. You can change some Diagramsettings in the sidebar. The black arrow is the representation of the gradient of the Orderfunktion.')

# import database
db = pd.read_feather(c.DBPATH)

# datapreparation for displaying the featurspace
[images_received, images_notreceived] = h.getfeaturespace(db)

# displaying properties
opacity = st.sidebar.slider('Opacity', 0.0, 1.0, 0.5)
size = st.sidebar.slider('Markersize',0,10,2)

# creating the plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(images_notreceived[0],images_notreceived[1], c = 'k', alpha=opacity, s=size) #beachte reihenfolge, da die ersten übermahlt werden.
ax.scatter(images_received[0], images_received[1], c= 'g', alpha=opacity, s=size)
plt.legend(['not received images', 'received images'])
plt.xlabel('size')
plt.ylabel('sharpness')

#[x1, x2] = h.getinterestingness(db)

#origin = [6.0,-6.0]
#ax.quiver(origin[0], origin[1], c.weight_size, c.weight_sharpness, scale=10)

#ax.plot(x1,x2)

# displaying the plot
st.pyplot(fig)

#--------------------------------------------------------------------
st.header('Calculation Details')

# Display information on the calculation of size
st.subheader('Size')
st.markdown('Size: The size is calculated ...')
st.markdown(r"$s=\sqrt{x^2+y^2}=1$")
st.code('scrabbing code and dispaly here')

# Display information on calculation of sharpness
st.subheader('Sharpness')
st.markdown('Sharpness: ...')
st.markdown(r'Formula for sharpness: $ \frac{1}{2} $')
st.code('scrabbin code and display here')

