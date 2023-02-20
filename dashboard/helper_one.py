import constants as c

import pandas as pd
import numpy as np
from skimage import io
import os.path
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import exists



def fake_img(number):
    '''
    input is the image counter and the database,
    from that it creates a fake image width all detected particles
    '''
    #Getting DB
    # db = pd.read_feather(c.DBPATH)
    
    # Getting Imagename
    imagename = 'Os7-S1 Camera'+ number

    # Filter the database according to the particles occuring in the image
    db_img = st.session_state['db'][st.session_state['db']['filename'] == imagename]

    # creating empty dummy image
    dummy_img = np.ones((c.IMAGESIZE_X, c.IMAGESIZE_Y), dtype=np.uint8)*c.BACKGROUNDCOLOR

    # loop over every entry and add it to the image
    db_img = db_img.reset_index()  # make sure indexes pair with number of rows

    for index, row in db_img.iterrows():
        # Get local Variables
        left = int(row['left'])
        right = int(row['right'])
        top = int(row['top'])
        bottom = int(row['bottom'])
        # size_pixelcount = row['size_pixelcount']
        # sharpness = row['sharpness']
        # interestingness = row['interestingness']
        order = row['order']
        
        width = right - left
        height = bottom - top
        
        # st.write([left, right, top, bottom])
        
        #load cropped image if exists
        cropped_imgpath = c.CROPPEDIMAGEPATH + '/' + str(order) + '_' + imagename + '.png'
        
        # load whole image path in case whole image got downloaded
        imagepath = c.CROPPEDIMAGEPATH + '/' + imagename + '.png'
        
        #offset = np.array((left, top))
        
        if os.path.exists(imagepath):
            img = io.imread(imagepath) 
        elif os.path.exists(cropped_imgpath):
            cropped_img = io.imread(cropped_imgpath)
            dummy_img[top : (top + height), left : (left + width)] = cropped_img
            # convert the dummy image-array to an image
            img = Image.fromarray(dummy_img, 'L')
        else:
            cropped_img = np.ones((height, width), dtype=np.uint8)
            dummy_img[top : (top + height), left : (left + width)] = cropped_img
            # convert the dummy image-array to an image
            img = Image.fromarray(dummy_img, 'L')
        
        
        
    #img.save('my.png')
    # img.show()
    return [img, db_img]

def getfeaturespace(db):
    # Getting features
    x = np.array(db['size_pixelcount'])
    y = np.array(db['sharpness'])

    # color by status if received on earth (green), else (blue)
    order = db['order']
    filename = db['filename']


    images_received_x = []
    images_received_y = []
    images_received =[]
    images_notreceived_x = []
    images_notreceived_y = []

    for i in range(0,len(x)):
        # ord = '%06d' % order[i]
        ord = str(order[i])
        pathtofile = c.CROPPEDIMAGEPATH + '/' + ord + '_' + filename[i] + '.png'
        file_exists = exists(pathtofile)
        if file_exists:
            # color.append('g')
            images_received_x.append(x[i])
            images_received_y.append(y[i])
        else:
            # color.append('k')
            images_notreceived_x.append(x[i])
            images_notreceived_y.append(y[i])
            
    images_received = [images_received_x, images_received_y]
    images_notreceived = [images_notreceived_x, images_notreceived_y]
    
    return images_received, images_notreceived
    
def plot_featurespace():

    # Fixing random state for reproducibility
    # Fixing random state for reproducibility
    # np.random.seed(19680801)


    db = pd.read_feather(c.DBPATH)
    # Getting features
    x = np.log(np.array(db['size_pixelcount']))
    y = np.log(np.array(db['sharpness']))

    # color by status if received on earth (green), else (blue)
    order = db['order']
    filename = db['filename']

    color = []
    for i in range(0,len(x)):
        # ord = '%06d' % order[i]
        ord = str(order[i])
        pathtofile = c.CROPPEDIMAGEPATH + '/' + ord + '_' + filename[i] + '.png'
        file_exists = exists(pathtofile)
        if file_exists:
            color.append('g')
        else:
            color.append('k')
            
    opacity = st.sidebar.slider('Opacity', 0.0, 1.0, 0.5)
    size = st.sidebar.slider('Markersize',0,10,2)

    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    # fig, ax = plt.subplots()

    # the scatter plot:
    ax.scatter(x, y, s= size, c = color, alpha = opacity)
    plt.xlabel('size')
    plt.ylabel('sharpness')
    
    
    st.pyplot(fig)
    


    
def fake_img_order(order):

    fn = order2image(order)
    
    fn = '%06d' % fn #converting to 6 digit string

    # get the dummy-image
    [img, db_img] = fake_img(fn)
    # display the dummy-image
    st.image(img)
    st.caption('Os7-S1 Camera' + str(fn))
    
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
    box = st.sidebar.checkbox("Box position and dimensions", False)
    if box:
        db_display = pd.concat([db_display, db_img['xpos'], db_img['ypos'], db_img['width'], db_img['height']], axis=1)
    
    # write the table corresponding to the image
    st.write(db_display)

            

    # display the dummy-image
    # cv2.imshow('Dummy-image: ' + imagename ,dummy_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #display the image
    # imagepath = '/home/auth/Documents/Projekte/aid_icaps/inputdata/MINIMAL_EXAMPLE/images/' + imagename + '.png'
    # orig_img = io.imread(cropped_imgpath)
    # cv2.imshow('Original: ' + imagename , orig_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # convert the dummy image-array to an image
    # img = Image.fromarray(dummy_img, 'L')

    
def order2imagename(order):
    # import Data
    db = pd.read_feather(c.DBPATH)

    # filter for order
    df = db[db['order'] == order]
    filename = df['filename'].tolist()[0]
    fn = int(filename.split('Camera')[1])
    return fn

def getinterestingness(db):
    grad_interestingness = [c.weight_size, c.weight_sharpness]
    # getting the ordering function to display
    x1 = [10.0,3.0] # size
    x2 = [-12,-6] # sharpness
    return x1, x2


# some minimal example functions ------------------------------------------------------
def test():
    st.write('hello')
    
def calc_interestingness(feature_sharpness, feature_size):
    interestingness = np.multiply(c.weight_sharpness * feature_sharpness, c.weight_size * feature_size)
    return interestingness
    
