
import matplotlib.pyplot as plt
import numpy as np
import os

real = 0.688109
img = 0.40148 
name = "image.jset"

def plot_at_idx(x):
    fig, ax = plt.subplots(1,1)
    full_img = np.concatenate((np.flip(np.flip(x, 0), 1), x), axis=0)
    ax.set_axis_off()
    plt.imshow(full_img)
    plt.title("Re{c}=" + str(real) + " Im{c}=" + str(img))
    plt.imsave('figure2.png', full_img)
    plt.show()

os.chdir("../data-generation/cpluplus_vViktor/")
os.system("make")
os.system("./fractals " + str(real) + " " + str(img) + " " + name)
os.system("make clean")
os.system("mv " + name + " ../../data-graphing/")
os.chdir("../../data-graphing/")

plot_at_idx(np.genfromtxt(name, delimiter=","))
