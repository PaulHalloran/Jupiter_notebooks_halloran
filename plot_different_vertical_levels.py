import numpy
import iris.quickplot as qplt
import matplotlib.pyplot as plt

def plot(depth,cube):
    depths = cube.coord('depth').points
    loc = numpy.where(depths > depth)[0][0]
    plt.figure(figsize=(10,12))
    qplt.contourf(cube[loc],20)
    plt.gca().coastlines()
    plt.title('depth = '+str(depth)+' m')
    plt.show()
