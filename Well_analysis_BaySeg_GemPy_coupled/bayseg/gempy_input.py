# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:45:46 2018

@author: Tobias Giesgen
"""

# data convertion to gempy
formation = clf.labels[-1]+1            # vector with final labeling including all wells 
                                        # labels + 1 because gempy doesn't like the zero as formation name

gempy = pd.DataFrame({'X': data.X ,'Y' : data.Y ,'Z' : data.Z ,'formation' : formation, 'borehole': data['Well Name']})

for k in range(0,len(gempy)-1):
    if gempy.loc[k,'formation'] == gempy.loc[k+1,'formation']:
        gempy = gempy.drop(k)
        
gempy.index = range(len(gempy)) 

for k in range(0,1 + len(set(list(gempy['formation'])))):
    gempy['formation'] = gempy['formation'].replace(to_replace = k, value = 'Layer%d' %(k))
    
for k in range(0,1 + len(set(list(gempy['formation'])))):
    gempy['formation'] = gempy['formation'].replace(to_replace = k, value = 'Layer%d' %(k))

gempy.to_csv('../data/Gempy_Simple_4_layer_90degrees.csv',index=False)