import cv2
import pdb
import os
import numpy as np
from tqdm import tqdm

for name in tqdm(os.listdir("./images")):
    vis0 = cv2.imread(f"./images/{name}")
    edge = cv2.imread(f"./edges/{name}",0)
    edge = edge>0
    #green = np.zeros_like(vis0)
    #green[edge>0,1] = 255
    #green1 = np.ones_like(vis0)*255
    #green1[edge>0,0] =0
    #green1[edge>0,2] =0
    green  = np.zeros_like(vis0)
    green1  = np.ones_like(vis0)*255
    white =  np.ones_like(vis0)*255
    vis = vis0/2
    vis = vis*(1-edge[...,None])+green1*edge[...,None]
    cv2.imwrite(f"./vis/{name}", vis)
    vis = vis0/2+np.ones_like(vis0)*128
    vis = vis*(1-edge[...,None])+green*edge[...,None]
    cv2.imwrite(f"./visb/{name}", vis)
