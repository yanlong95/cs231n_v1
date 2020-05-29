# -*- coding: utf-8 -*-
"""
Retrieve building tag based on coordinates given falling into building footprint
Created on Thu May 28 15:12:00 2020

@author: jack_minimonster
"""
import pandas as pd
import overpy
import numpy as np
from tqdm import tqdm



# locate file path and information #
path = 'C:/Users/jack_minimonster/Documents/231n_dataset/BIC_GSV/city_scale_GSV_images/'
filename = 'boston_latlon.txt'
# filename = 'calgary_latlon.txt'
# filename = 'toronto_latlon.txt'

# convert txt file into dataframe #
df_lat_lon = pd.read_csv(path + filename, sep = '[/, :, ,]', header=None, 
                         names=['0', '1', 'imgname', 'lat', 'lon'], dtype='str')
df_lat_lon = df_lat_lon[['imgname', 'lat', 'lon']] # data format: image name, lat, lon all in string
df_lat_lon['tag'] = "" # make a new tag column

# enabling overpass api #
api = overpy.Overpass()

with tqdm(total=len(df_lat_lon)) as t:
    for index, row in df_lat_lon.iterrows():
        coord = row[['lat', 'lon']] # string in series
        radius = 10
        # see overpass api --> [around: radius, target_pt_lat, target_pt_lon]
        input_str = "way['building'](around: " + str(radius) + ", " + coord.iloc[0] + "," + coord.iloc[1] + ");out;"
        result = api.query(input_str)
        
        # if no target in radius, increase radius until find one
        while len(result.ways) == 0:
            radius += 5
            input_str = "way['building'](around: " + str(radius) + ", " + coord.iloc[0] + "," + coord.iloc[1] + ");out;"
            result = api.query(input_str) # requery
        
        # find building tag in way element
        building_category = result.ways[0].tags.get('building')
        
        # append tag to dataframe
        row['tag'] = building_category
        
        # saving at every 500 iteration
        if index % 500 == 0 and index != 0:
            print("...saving checkpoint...")
            df_lat_lon.to_csv(filename[:-4] + '-category_labelled.csv')
            
        t.update()
              
print("...saving final result...")
df_lat_lon.to_csv(filename[:-4] + '-category_labelled.csv')
print("saving done!")
