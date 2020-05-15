# -*- coding: utf-8 -*-
"""
OSM file viewer
Obtain OSM building nodal coordinate and save in framed structure
Calculate centroid of the building footprint polygon and export as csv based on building id

docs: https://docs.osmcode.org/pyosmium/latest/index.html

Required package: panda, matplotlib, osmium, csv, random

Created on Tue May 5 11:05:21 2020

@author: Jack Li
"""

import osmium as osm
import pandas as pd
import matplotlib.pyplot as plt
import random
import csv

# define handler for different elements in OSM file
# @param osm.
class OsmHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_data_node = []
        self.osm_data_way = []
        self.osm_data_relation = []

    def node(self, n):
        self.osm_data_node.append(["node",
                              n.id,
                              n.visible,
                              n.location.y,
                              n.location.x,
                              len(n.tags)
                              ])

    def way(self, w):
        
        # only check building with these tags on
        category_filter = ['apartments', 
                           'church', 
                           'garage', 
                           'house', 'detached', 'bungalow', 'semidetached_house', 'villa',
                           'industrial', 'warehouse',
                           'office', 
                           'retail',
                           'hotel',
                           'roof',
                           #'yes' # this is selected only if the filter can't generate enough building
                           ]
        
        if w.is_closed() and (w.tags.get('building') in category_filter): # check if is building footprint polygon
             for anode in w.nodes:
                 self.osm_data_way.append(["way",
                                   w.id,                 # building id
                                   anode.ref,            # edge node id
                                   anode.location.y,     # edge node lat
                                   anode.location.x,     # edge node lon
                                   len(w.tags),           # num of tags
                                   w.tags.get('building')         
                                   ])

    # def relation(self, r):
    #     self.osm_data_relation.append(["relation",
    #                           r.id,
    #                           r.visible,
                              
    #                           len(r.tags)
    #                           ])

# find centroid of a non self-interacting closed polygon by n nodes
# @ param pt_x / y_list: list of lat and lon values
# @ return c_x / c_y: centroid coordinate of the polygon
# @ ref https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
def find_centroid_polygon(pt_x_list, pt_y_list):
    pt1_x, pt2_x, pt1_y, pt2_y, f, A, f_x, f_y = 0, 0, 0, 0, 0, 0, 0, 0
    
    for i in range(0, len(pt_x_list)-1):
        pt1_x = pt_x_list[i]
        pt1_y = pt_y_list[i]
        pt2_x = pt_x_list[i+1]
        pt2_y = pt_y_list[i+1]
        f = pt1_x * pt2_y - pt2_x * pt1_y
        A += 0.5 * f
        
        f_x += (pt1_x + pt2_x) * f
        f_y += (pt1_y + pt2_y) * f
        
    c_x = int(f_x / (A * 6))
    c_y = int(f_y / (A * 6))
    return c_x, c_y

# generate dataframe containing building id and centroid coordinate
# @ param df: raw dataframe from osm file
# @ return polygon_centroid_df: dataframe with building id and centroid coordinate
def generate_centroid_dataframe(df):
    node_lat_list = []
    node_lon_list = []
    polygon_list = {}
    checker_pre = 0
    checker = 0
    # group node lat and lon based on polygon id
    for index, row in df_osm_way.iterrows():
        checker = row['id']
        if (checker_pre == checker):
            node_lat_list.append(row['node_lat'])
            node_lon_list.append(row['node_lon'])
            polygon_list[row['id']] = [node_lat_list, node_lon_list]
        else:
            node_lat_list = []
            node_lon_list = []
            node_lat_list.append(row['node_lat'])
            node_lon_list.append(row['node_lon'])
            polygon_list[row['id']] = [node_lat_list, node_lon_list]  
        checker_pre = checker

    # calculate polygon centroid and save it as dictionary per building id
    polygon_centroid_list = {}
    for key in polygon_list:
        c_x, c_y = find_centroid_polygon(polygon_list[key][0], polygon_list[key][1])
        polygon_centroid_list[key] = [c_x, c_y]
    
    polygon_centroid_df = pd.DataFrame.from_dict(polygon_centroid_list, orient='index')
    
    return polygon_centroid_df


######################## main ############################

osmhandler = OsmHandler()
# scan the input file and fills the handler list accordingly
PATH = "C:/Users/jack_minimonster/Documents/231n_dataset/BIC_GSV/data_query/osm_file/"
filename = "georgia.osm.pbf"

osmhandler.apply_file(PATH + filename, locations=True) # set location to True to get coordinate

# extract building footprint information from osm and sort based on building id
data_colnames_way = ['type', 'id',
                  'edge_node_id', 'node_lat', 'node_lon', 'ntags', 'building type']
df_osm_way = pd.DataFrame(osmhandler.osm_data_way, columns=data_colnames_way)
df_osm_way = df_osm_way.sort_values(by=['type', 'id'])

df_osm_grouped = df_osm_way.groupby('building type')

for group_name, group_value in df_osm_grouped:
    df_osm_building_group = df_osm_grouped.get_group(group_name)
    polygon_centroid_df = generate_centroid_dataframe(df_osm_building_group)

    # randomly pick certain number of examples from the full building list
    polygon_centroid_df_sample = polygon_centroid_df.sample(n=10)
    csvname = filename[:-8] + '-' + str(group_name)

    # # save the centroid lat lon information as csv file
    print('...saving csv - ' + csvname + '...')
    polygon_centroid_df_sample.to_csv(csvname + '.csv')

# csv_filename = 'building_centroid_coord.csv'
# with open(csv_filename, 'w') as csv_file:
#     for key in polygon_centroid_list_sample.keys():
#         csv_file.write("%s,%s,%s\n"%(key,polygon_centroid_list_sample[key][0],
#                                      polygon_centroid_list_sample[key][1]))

# # output csv file in format 'building_id, [centroid lat x 1e7, centroid lon x 1e7]
