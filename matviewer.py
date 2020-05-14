# -*- coding: utf-8 -*-
"""
mat file viewer, check lat lon information of the test region data
convert mat array into np array

Created on Mon May  4 14:27:20 2020

@author: jack_minimonster
"""


import numpy as np
import scipy.io

PATH = 'C:/Users/jack_minimonster/Documents/231n_dataset/BIC_GSV/vancouver_test_region/'
filename = 'building_instance_lat_lon.mat'
filename2 = 'building_mask_label.mat'

file_instance = scipy.io.loadmat(PATH + filename)
file_mask = scipy.io.loadmat(PATH + filename2)

building_instance = np.array(file_instance['building_instance'])
building_mask = np.array(file_mask['labeled_building_mask'])

print(building_instance[0,2].shape)
