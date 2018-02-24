"""
This file provides a function for visualizing an audiosegment
in the time domain.
"""
import importlib.util
__spec = importlib.util.spec_from_file_location("audiosegment", "../audiosegment.py")
asg = importlib.util.module_from_spec(__spec)
__spec.loader.exec_module(asg)
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

VIS_MS = 30000 # The number of ms to use in visualizing stuff

def visualize(seg, title=""):
    plt.plot(seg.to_numpy_array())
    plt.title(title)
    plt.show()
    plt.clf()

