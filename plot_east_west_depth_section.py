import numpy
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris

def plot(lon_west,lon_east,lat_south,lat_north,cube):
    cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
    cube = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
    try:
        cube.coord('latitude').guess_bounds()
    except:
        pass
    try:
        cube.coord('longitude').guess_bounds()
    except:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    section_avg_cube = cube.collapsed(['latitude'], iris.analysis.MEAN, weights=grid_areas)
    plt.figure(figsize=(10,12))
    qplt.contourf(section_avg_cube,20)
    plt.show()
