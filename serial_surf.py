from __future__ import print_function
import numpy as np
import mahotas as mh
from mahotas.features import surf
from pylab import *
import sys
from os import path

images = ['./1.JPG', './2.JPG', './3.JPG']
#images = ['./3.JPG']

debug = True if len(sys.argv) < 2 else False
repeat = 1 if debug else 20

for image in images:
    f = mh.imread(image, as_grey=True)
    f = f.astype(np.uint8)
    for i in range(repeat):
        spoints = surf.surf(f = f, nr_octaves=4, nr_scales=6, initial_step_size=2, threshold=0.99, max_points=24, descriptor_only=False)

    if debug:
        print("Nr points:", len(spoints))

        values = np.zeros(100)
        colors = np.array([(255,0,0)])
    
        f2 = surf.show_surf(f, spoints[:100], values, colors)
        imshow(f2)
        show()
