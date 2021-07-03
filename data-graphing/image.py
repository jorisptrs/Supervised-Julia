
import matplotlib.pyplot as plt
import numpy as np
import os

real = 0.23420172929763794
img = -0.5451204180717468
name = "image.jset"
size = 2000

def plot_at_idx(x):
    fig, ax = plt.subplots(1,1)
    full_img = np.concatenate((np.flip(np.flip(x, 0), 1), x), axis=0)
    ax.set_axis_off()
    plt.imshow(full_img, cmap=plt.cm.gray_r)
    plt.title("Re{c}=" + str(real) + " Im{c}=" + str(img))
    plt.imsave('c' + str(round(real, 3)) + " " + str(round(img, 3)) + "i.pdf", full_img, cmap=plt.cm.gray_r)
    plt.show()

os.chdir("../data-generation/cpluplus_vViktor/")
os.system("make")
os.system("./fractals " + str(real) + " " + str(img) + " " + name + " " + str(size))
os.system("make clean")
os.system("mv " + name + " ../../data-graphing/")
os.chdir("../../data-graphing/")

plot_at_idx(np.genfromtxt(name, delimiter=","))
