# -*- coding: utf-8 -*-
"""
Retrieve Street View Image (512 x 512) Through Google Street View API
doc: https://rrwen.github.io/google_streetview/index.html
    https://pypi.org/project/google-streetview/


Required package: google_streetview, csv, decimal

Created on Tue May 12 16:38:31 2020

@author: jack_minimonster
"""

import google_streetview.api as gsvapi
import google_streetview.helpers as gsvhelpers
import csv
#from decimal import Decimal
    
gsv_api_key = 'AIzaSyDlpR8QBNNMHFXFoQol0zVbMpt_aLrkdeM' # insert your own google street view static API

csv_filename = 'centroid_csv_file/georgia_apartment.csv'
# csv file is in format of 'building id, centroid lat, centroid lon'

location_str = ''
with open(csv_filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # skip the header
    for row in csv_reader:
        centroid_lat = str(row[1])
        centroid_lon = str(row[2])
        # the coord is saved as x 1e7 integer to preserve precision
        # to obtain the actual coord needs to divide by 1e7
        location_str += centroid_lat[:-7] + '.' + centroid_lat[-7:] + ',' + centroid_lon[:-7] + '.' + centroid_lon[-7:] + ';'

print('...reading parameters...')
params = {
  'size': '512x512', # max 640x640 pixels
  'location': location_str[:-1],
  #'heading': '0;30;60;90;120;150;180;210;240;270;300;330', # if don't specify heading it will face the coordinate point
  'pitch': '10',
  'key': gsv_api_key,
  'source': 'outdoor'
}

# Create a results object
street_view_list = gsvhelpers.api_list(params)
print('...data querying from API...')
results = gsvapi.results(street_view_list)

# Download images to directory 'downloads'
# results.preview()
print('...saving images...')
results.download_links(csv_filename[17:-4])
print('...saving metadata...')
results.save_metadata('street_view_metadata.json')
