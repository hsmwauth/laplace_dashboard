'''
Following questions should be anwered on this page:
    * selecting the experiment run
    * displaying rough information on this experiment run.
        * start timedate
        * end timedate
        * running/transferring data/finised (status)
'''

# from PIL import Image
import streamlit as st
import helper_one as h
import constants as c
import pandas as pd
import numpy as np

# Defining the dataframe as session state/ global Variable that it doesnt need to be be rerun everytime/-page
if 'db' not in st.session_state:
    st.session_state['db'] = pd.read_feather(c.DBPATH)
    
if 'frame' not in st.session_state:
    st.session_state['frame'] = 'Os7-S1 000001'
    
if 'order' not in st.session_state:
    st.session_state['order'] = 1
    
if 'imagelist' not in st.session_state:
    st.session_state['imagelist'] = []
    


st.title('Experiment - ID xxxxxxy')
st.info('On this page the main specs of the exeriments should be displayed')

st.header('Experiment specification')
st.write('Hier wird vom experiment-setupfile alles wichtige gescrapped.')
st.warning('Stand: Warte auf json-file um den scrabber zu schreiben', icon="⚠️") 
st.markdown('Time: 2024-12-3: 11:30 to 2024-12-3: 12:31')
st.markdown('n_images: xxxxxxxy')
st.markdown('...')
st.markdown('Diagramm with magjor changes in current etc.')

st.write('session states:')
st.write('Frame: ' + str(st.session_state['frame']))
st.write('Order: ' + str(st.session_state['order']))

st.write('Imagelist: ')
clearimagelist = st.button('clear the imagelist')

if clearimagelist:
    st.session_state['imagelist'] = []
    
st.write(st.session_state['imagelist'])

imagelist = np.asarray(st.session_state['imagelist'])

path2save = st.text_input('Path to save the imagelist to ...', value='./imagelist.csv')
save = st.button('save the imagelist to defined path ...')
if save:
    imagelist.tofile(path2save, sep=',')
    st.success('Wrote list int file ' + path2save)
