# -*- coding: utf-8 -*-
"""

OSM file viewer using osmium
docs: https://docs.osmcode.org/pyosmium/latest/index.html

Required package: numpy, panda, matplotlib, osmium

Created on Tue May  5 11:05:21 2020

@author: jack_minimonster
"""


import osmium as osm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# define handler for different elements in OSM file
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
       if w.is_closed() and ("building" in w.tags): # check if is building footprint polygon
            for anode in w.nodes:
                self.osm_data_way.append(["way",
                                  w.id,
                                  anode.ref,
                                  anode.location.y,
                                  anode.location.x,
                                  len(w.tags)
                                  ])

    # def relation(self, r):
    #     self.osm_data_relation.append(["relation",
    #                           r.id,
    #                           r.visible,
                              
    #                           len(r.tags)
    #                           ])
    

osmhandler = OsmHandler()
# scan the input file and fills the handler list accordingly
PATH = "C:/Users/jack_minimonster/Documents/231n_dataset/BIC_GSV/vancouver_satillate/"
filename = "vancouver_test_region_osm.osm"

osmhandler.apply_file(PATH + filename, locations=True) # set location to True to get coordinate

# data_colnames_node = ['type', 'id', 'visible',
#                   'lat', 'lon', 'ntags']
# df_osm_node = pd.DataFrame(osmhandler.osm_data_node, columns=data_colnames_node)
# df_osm_node = df_osm_node.sort_values(by=['type', 'id'])

data_colnames_way = ['type', 'id',
                  'edge_node_id', 'node_lat', 'node_lon', 'ntags']
df_osm_way = pd.DataFrame(osmhandler.osm_data_way, columns=data_colnames_way)
df_osm_way = df_osm_way.sort_values(by=['type', 'id'])


# plt.plot(df_osm_way['node_lon'], df_osm_way['node_lat'], 'o')
# plt.title('Vancover Test Region')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.axis('equal')
# plt.show()











