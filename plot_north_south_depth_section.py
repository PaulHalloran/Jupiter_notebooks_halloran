import numpy
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris


def plot(lon_west,lon_east,lat_south,lat_north,cube):
    cube_region_tmp = cube.intersection(longitude=(lon_west, lon_east))
    cube_region = cube_region_tmp.intersection(latitude=(lat_south, lat_north))
    try:
        cube_region.coord('latitude').guess_bounds()
    except:
        pass
    try:
        cube_region.coord('longitude').guess_bounds()
    except:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube_region)
    section_avg_cube = cube_region.collapsed(['longitude'], iris.analysis.MEAN, weights=grid_areas)
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    qplt.contourf(cube[0],20)
    plt.plot([((lon_east-lon_west)/2.0)+lon_west,((lon_east-lon_west)/2.0)+lon_west],[lat_south,lat_north])
    plt.gca().coastlines()
    plt.subplot(122)
    qplt.contourf(section_avg_cube,20)
    plt.show()
