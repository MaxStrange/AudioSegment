"""
This file provides a function for visualizing an audiosegment
in the time domain.
"""
import sys
sys.path.insert(0, '../')
import audiosegment as asg
import os
import platform
if os.environ.get('DISPLAY', False):
    import matplotlib
    if platform.system() != "Windows":
        matplotlib.use('QT5Agg')
    import matplotlib.pyplot as plt

VIS_MS = 30000 # The number of ms to use in visualizing stuff

def visualize(seg, title=""):
    if os.environ.get('DISPLAY', False):
        plt.plot(seg.to_numpy_array())
        plt.title(title)
        plt.show()
        plt.clf()
