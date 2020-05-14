# -*- coding: utf-8 -*-
"""
Retrieve Street View Image (512 x 512) Through Google Street View API
doc: https://rrwen.github.io/google_streetview/index.html


Required package: numpy, panda, matplotlib, google_streetview

Created on Tue May 12 16:38:31 2020

@author: jack_minimonster
"""

import numpy as np
import matplotlib as plt
import google_streetview.api as gsvapi
import google_streetview.helpers as gsvhelpers

gsv_api_key = 'AIzaSyDHDDb152DY6d7bHjz-DEbrNOptudfad6U' 
# can be accessed through street view static API

params = {
  'size': '512x512', # max 640x640 pixels
  'location': '49.2845176, -123.1301644',
  'heading': '0;30;60;90;120;150;180;210;240;270;300;330',
  'pitch': '10',
  'key': gsv_api_key
}

# Create a results object
street_view_list = gsvhelpers.api_list(params)
results = gsvapi.results(street_view_list)

# Download images to directory 'downloads'
results.preview()
results.download_links('street_view_downloads')
results.save_metadata('street_view_metadata.json')