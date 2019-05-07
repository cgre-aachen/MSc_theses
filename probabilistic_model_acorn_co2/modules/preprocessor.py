"""
    This file is part of AMTK - the Acorn mesh toolkit, the companion
    software package to the Master's Thesis "The value of probabilistic
    geological modeling: Application to the Acorn CO2 storage site"
    of Marco van Veen

    AMTK is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AMTK is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with AMTK.  If not, see <http://www.gnu.org/licenses/>.

@author: Marco van Veen
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import norm, probplot
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path



def rotate_surface(surface, angle, pivot=(0, 0)):
    '''Rotates a given x,y,z surface in 2D space counterclockwise by a given angle around a point of origin (pivot point).

    Arguments:
        surface(pandas.DataFrame): Vertices / data points with 'X' and 'Y' columns.
        angle(float): Rotation angle in degrees for counterclockwise rotation. Clockwise rotation by negative values.
        pivot(float, float): Optional point of origin (default (0,0)).

    Returns(pandas.DataFrame): Rotated set of vertices (deepcopy).
    '''

    output = surface.copy(deep=True)

    a = np.cos(np.radians(angle))
    b = np.sin(np.radians(angle))

    output.X = a * (surface.X - pivot[0]) - b * (surface.Y - pivot[1]) + pivot[0]
    output.Y = b * (surface.X - pivot[0]) + a * (surface.Y - pivot[1]) + pivot[1]

    return output


def rotate_mesh(vertices, angle, pivot=(0, 0)):
    '''Rotates a given x,y,z surface in 2D space counterclockwise by a given angle around a point of origin (pivot point).

    Arguments:
        vertices(numpy.array): Vertices / data points with x and y positions
        angle(float): Rotation angle in degrees for counterclockwise rotation. Clockwise rotation by negative values.
        pivot(float, float): Optional point of origin (default (0,0)).

    Returns(pandas.DataFrame): Rotated set of vertices (deepcopy).
    '''

    output = np.copy(vertices)

    a = np.cos(np.radians(angle))
    b = np.sin(np.radians(angle))

    output[:,0] = a * (vertices[:,0] - pivot[0]) - b * (vertices[:,1] - pivot[1]) + pivot[0]
    output[:,1] = b * (vertices[:,0] - pivot[0]) + a * (vertices[:,1] - pivot[1]) + pivot[1]

    return output


def triangulate(vertices, dim=2):
    '''
    Creates a pandas dataframe representing the simplices of a triangle mesh, derived from vertices with the Delaunay triangulation.

    Arguments:
        vertices: pandas DataFrame of xyz coordinates.
        dim (int): number of dimensions used for triangulation - can be used to disregard the z dimension (default 2).
    Returns:
        simplices: pandas DataFrame of simplices from Delaunay triangulation
    '''

    simplices = Delaunay(vertices.iloc[:, :dim].values).simplices

    # TODO: If 3D triangulation is used, dataframe must be adjusted accordingly
    simplices = pd.DataFrame(data=simplices, columns=['A', 'B', 'C'])

    return simplices


def find_boundary_clockwise(vertices, simplices):
    # takes numpy.array of vertices and simplices of a shape
    # returns an ordered numpy.array (clockwise or counterclockwise) of vertices of the shape boundary

    # extract edges from faces
    all_edges = np.concatenate((simplices[:, [0, 1]],
                                simplices[:, [1, 2]],
                                simplices[:, [0, 2]])
                               )

    # sort the edge pairs, so for an edge pair [a,b] always applies: a < b
    sorted_edges = np.sort(all_edges, axis=-1)

    # find edge pairs that only occur once, so they are only part of one face/triangle
    unique, counts = np.unique(sorted_edges, return_counts=True, axis=0)
    edges = unique[counts == 1]

    # initiate loop
    node_list = []  # ordered list of nodes

    # just start with first node index
    start_stop = edges[0, 0]
    node_list.append(start_stop)

    row = 0
    column = 1
    node = edges[row, column]

    while node != start_stop:
        # add current node, because not yet start_stop
        node_list.append(node)

        # remove current edge
        edges = np.delete(edges, row, axis=0)

        # find next position of current node
        row, column = np.argwhere(edges == node)[0]

        # get new node
        node = edges[row, 1] if column == 0 else edges[row, 0]

    return vertices[node_list]


def clip_boundary(vertices, simplices, boundary_vertices, radius=0):
    '''
    vertices: numpy arrays
    simplices: numpy array OR None
    boundary_vertices: numpy array: (counter-)clockwise sorted array of vertices constituting the boundary
    '''

    boundary_path = Path(boundary_vertices[:, 0:2])
    mask = boundary_path.contains_points(vertices[:, 0:2], radius=radius)

    # clip vertices, reset index, but keep old index
    clip_ver = vertices[mask]

    if simplices is None:
        return clip_ver

    # check whether each simplex has all masked vertices or not
    # https://stackoverflow.com/questions/53376827/selecting-faces-of-a-mesh-based-on-vertices-coordinates-in-numpy
    sim_mask = np.all(mask[simplices], axis=1)
    # select resulting simplices
    clip_sim = simplices[sim_mask]

    # create a map from old to new indices
    # https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    old_idx = np.nonzero(mask)[0]
    new_idx = np.arange(old_idx.shape[0])
    mp = np.arange(0, old_idx.max() + 1)
    mp[old_idx] = new_idx

    # map old vertex indices to new ones inside the simplices
    clip_sim = mp[clip_sim]

    return clip_ver, clip_sim


class FaceFilter(object):
    """FaceFilter class to identify and clip unwanted faces/simplices from a (convex) Delaunay triangulation.

    Parameters:
        vertices (numpy.array[:,3]): Array of nodes.
        simplices (numoy.array[:,3]): Array of faces/triangles.
        dimension (int): Integer of the dimension, the distances are calculated in.
    """

    def __init__(self, vertices, simplices, dimension=2):

        self.ver = vertices
        self.sim = simplices
        self.dim = dimension

        self.dist = self._edge_distances()


    def _edge_distances(self):
        # calculating edge distances for each simplex
        dist = np.zeros_like(self.sim, dtype='float32')

        for i in np.arange(dist.shape[0]):
            dist[i] = [np.linalg.norm(self.ver[self.sim[i, 0], :self.dim] - self.ver[self.sim[i, 1], :self.dim]),
                       np.linalg.norm(self.ver[self.sim[i, 0], :self.dim] - self.ver[self.sim[i, 2], :self.dim]),
                       np.linalg.norm(self.ver[self.sim[i, 1], :self.dim] - self.ver[self.sim[i, 2], :self.dim])]

        return dist

    def hist(self, figsize=(16,8), range=(0, 500), log=True, bins=100):
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.dist.ravel(), log=log, bins=bins, range=range)
        if self.dim == 2:
            ax.set_xlabel('Edge lengths in xy-direction [m]')
        elif self.dim == 3:
            ax.set_xlabel('Edge lengths in xyz-direction [m]')
        else:
            ax.set_xlabel('Edge distances')
        ax.set_ylabel('Frequency')
        plt.grid(linestyle='dotted')

    def edge_cutoff(self, cut_off):
        # cut_off: distance, where all distances greater are clipped
        # returns the clipped simplices

        mask = np.where(self.dist <= cut_off, True, False).prod(axis=1).astype(bool)
        return self.sim[mask]


def datashade_surfaces(xarray_list, name_list, domain_list=None, wells=None, wells_label='Wells', cols=1, plot_size=(750, 400)):
    ''' Plot several surfaces with dynamic datashader

    xarray_list: list of xarrays to plot
    name_list: list of str
    domain_list: list of int to choose colormaps (0: depth, 1: time)
    wells: optional pandas.DataFrame of welltops, added to all plots
    wells_label: optional str to label the well locations
    cols: number of columns
    plot_size: Tuple of int: The individual size of every subplot
    '''

    import holoviews as hv
    from holoviews.operation.datashader import datashade
    from bokeh.palettes import viridis, inferno, RdBu, Spectral
    from bokeh.models import NumeralTickFormatter#, WheelZoomTool
    hv.extension('bokeh')

    # styling HoloView plots
    def axes_formatter(plot, element):
        plot.handles['xaxis'].formatter = NumeralTickFormatter(format='7')
        plot.handles['yaxis'].formatter = NumeralTickFormatter(format='7')
        # plot.state.toolbar.active_scroll = plot.state.tools[2]
        # plot.handles['active_scroll'] = 'wheel_zoom'
        # plot.state.toolbar.active_scroll = WheelZoomTool()

    # domain specific cmaps
    cmaps = [viridis(256), inferno(256)]
    if domain_list is None:
        domain_list = [0] * len(xarray_list)

    # to modify axes, modify in any case the RGB instances of HoloView with the plot arguments!
    opts = {'RGB': {'style': dict(),
                    'plot': dict(width=plot_size[0], height=plot_size[1], aspect='square', show_grid=True,
                                 colorbar=True, finalize_hooks=[axes_formatter])},
            'Points': {'style': dict(color='black', marker='*', size=10),
                       'plot': dict(tools=['hover'])}
            }

    plot_list = []
    for i, x in enumerate(xarray_list):
        image = hv.Image(x, ['X', 'Y'], 'Z')
        plot = (datashade(image, cmap=cmaps[domain_list[i]]))

        if wells is not None:
            points = hv.Points(wells, ['X', 'Y'], vdims=['Z', 'well', 'formation'], label=wells_label)
            plot = plot * points

        plot = plot.relabel(name_list[i])
        # maybe for holoviews 1.11
        # plot = plot.options(active_tools=['wheel_zoom'])
        plot_list.append(plot)

    layout = hv.Layout(plot_list).cols(cols).opts(opts)
    return layout


def surface_stats(surface_list, name_list, domain='Depth'):
    '''Plot pdf, cdf (data based and artificial) and probability plot

    surface_list: list of pandas.DataFrames with ['Z'] columns.
    name_list: list of name str according to surface list.
    domain: str (e.g. time or depth).
    '''

    # every even color is a pale version of every odd one
    colors = plt.cm.get_cmap('tab20').colors

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for s, surface in enumerate(surface_list):
        z = surface['Z']

        ax[0].hist(z, bins=30, density=True, cumulative=False, alpha=1, color=colors[2 * s + 1], label=name_list[s])
        ax[0].plot(np.arange(np.min(z), np.max(z), 1),
                   norm.pdf(np.arange(np.min(z), np.max(z), 1), np.mean(z), np.std(z)),
                   color=colors[2 * s],
                   label=None
                   )

        ax[1].hist(z, bins=30, density=True, cumulative=True, alpha=1, color=colors[2 * s + 1], label=None)
        ax[1].plot(np.arange(np.min(z), np.max(z), 1),
                   norm.cdf(np.arange(np.min(z), np.max(z), 1), np.mean(z), np.std(z)),
                   color=colors[2 * s],
                   label=None
                   )

        probplot(z, dist='norm', plot=ax[2])
        ax[2].get_lines()[2 * s].set_markerfacecolor(colors[2 * s + 1])
        ax[2].get_lines()[2 * s].set_markeredgecolor(colors[2 * s + 1])
        # ax[2].get_lines()[2*s].set_alpha(0.5)
        ax[2].get_lines()[2 * s].set_marker('.')
        ax[2].get_lines()[2 * s + 1].set_color(colors[2 * s])

    ax[0].set_title('PDF')
    ax[0].set_xlabel(domain)
    ax[0].set_ylabel('Probability')
    ax[0].grid()

    ax[1].set_title('CDF')
    ax[1].set_xlabel(domain)
    ax[1].set_ylabel('Cumulative probability')
    ax[1].grid()

    ax[2].set_title('Probability plot')
    # ax[2].set_xlabel()
    ax[2].set_ylabel('Ordered values')
    ax[2].grid()

    fig.legend(loc='lower left',
               bbox_to_anchor=(1.01, 0.1),
               ncol=1, borderaxespad=0, frameon=False)

    plt.tight_layout()