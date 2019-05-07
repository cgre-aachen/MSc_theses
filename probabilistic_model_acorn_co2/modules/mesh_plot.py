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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import holoviews as hv
from bokeh.models import NumeralTickFormatter


def plot_mesh_error_legacy(points_list, error_list, names, figsize=(16,8), ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    ax.scatter(points_list, error_list, label=names)
    ax.plot(points_list, error_list)

    ax.set_xlim(max(points_list) + 100, 0)
    ax.set_xlabel('Number of points')
    ax.set_ylabel('Divergence / Euclidean Hausdorff Distance')
    ax.grid(True)
    ax.legend()


def plot_mesh_error_legacy_2(evaluation, sample=0, figsize=(16,8), ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    # plot unsampled Hausdorff distance
    if sample == 0 or sample == 2:
        ax.plot(evaluation['size'].values, evaluation['hausdorff'].values,
                marker='o', linestyle='--', label=evaluation.loc[0, 'description'])

    # plot sampled Hausdorff distance
    if sample == 1 or sample == 2:
        ax.plot(evaluation['size'].values, evaluation['sample_hausdorff'].values,
                marker='o', linestyle='-', label=(evaluation.loc[0, 'description'] + ''))

    ax.set_xlim(evaluation['size'].max() + 100, 0)
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Approximation error (Euclidean Hausdorff distance)')
    ax.grid(True)
    ax.legend()

def plot_mesh_error(evaluation, figsize=(16, 8), ax=None, measure='hausdorff'):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = ax

        # plot sampled Hausdorff distance
        ax.plot(evaluation['size'].values, evaluation[measure].values,
                    marker='o', linestyle='-', label=(evaluation.loc[0, 'description'] + ''))

        ax.set_xlim(evaluation['size'].max() + 100, 0)
        ax.set_xlabel('Number of vertices')
        ax.set_ylabel('Approximation error (Euclidean Hausdorff distance) [m]')

        ax.grid(True, linestyle='dotted')
        ax.legend()


def plot_vertices_2d(vertices, title=None, step=1, colored=False, figsize=(16,8), ax=None, xaxis=True, yaxis=True, colorbar=False, unit='', **kwargs):
    """Plot point set / vertices (pandas DataFrame) in 2D.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax

    if colored:
        plot = ax.scatter(vertices.values[::step, 0], vertices.values[::step, 1], c=vertices.values[::step, 2], **kwargs)
    else:
        plot = ax.scatter(vertices.values[::step, 0], vertices.values[::step, 1], **kwargs)

    ax.grid(True, linestyle='dotted')

    if title is not None:
        ax.set_title(title, size='16')

    if xaxis:
        ax.set_xlabel('X')
    else:
        ax.xaxis.set_ticklabels([])

    if yaxis:
        ax.set_ylabel('Y')
    else:
        ax.yaxis.set_ticklabels([])

    if colorbar:
        #fig.tight_layout()
        cbar = plt.colorbar(plot, ax=ax, pad=0.02)
        cbar.set_label(unit, rotation=270, labelpad=12)
        

def plot_vertices_3d(vertices, title=None, step=1, colored=False, colordimension=None, figsize=(16,8), ax=None, viewpoint=(30, -60), **kwargs):
    """Plot a point set / vertices (pandas DataFrame) in 3D.
    """

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.elev = viewpoint[0]
        ax.azim = viewpoint[1]
    else:
        ax = ax

    if colored:
        ax.scatter(vertices.values[::step, 0], vertices.values[::step, 1], vertices.values[::step, 2],
                   c=vertices.values[::step, 2], cmap='viridis', **kwargs)
    else:
        ax.scatter(vertices.values[::step, 0], vertices.values[::step, 1], vertices.values[::step, 2],
                   c=colordimension, cmap=plt.cm.nipy_spectral, **kwargs)

    if title is not None:
        ax.set_title(title, size='16')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_mesh_2d_legacy(vertices, simplices, title=None, colored=False):
    """Plot set of vertices and simplices (pandas DataFrames) interactively in 3D.
    """

    hv.extension('matplotlib')

    if colored:
        edge_color = 'Z'
    else:
        edge_color = ''

    if title is None:
        title = ''

    # styling plot
    options = {'TriMesh': {'style': dict(cmap='viridis',
                                         node_marker='o'
                                         ),
                           'plot': dict(fig_inches=(16, 8),
                                        aspect=2,
                                        show_grid=True,
                                        edge_color_index=edge_color,
                                        filled=colored,
                                        fontsize={'title': 16}
                                        )
                           }
               }

    trimesh = hv.TriMesh((simplices, vertices), label=title).opts(options)

    return trimesh


def plot_mesh_2d(vertices, simplices, title=None, colored=False, figsize=(16, 8), ax=None, **kwargs):
    """Plot set of vertices and simplices (pandas DataFrames) interactively in 3D.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    ax.triplot(vertices.values[:, 0], vertices.values[:, 1], simplices.values,
               color='k',
               marker='.',
               linestyle='-'
               )

    if colored:
        ax.tricontourf(vertices.values[:, 0], vertices.values[:, 1], simplices.values, vertices.values[:, 2])

    if title is not None:
        ax.set_title(title, size='16')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def bokeh_axes_formatter(plot, element):
    # helper function for bokeh and holoview
    # format Bokeh tick labels in HoloView via finalize_hooks

    plot.handles['xaxis'].formatter = NumeralTickFormatter(format='7')
    plot.handles['yaxis'].formatter = NumeralTickFormatter(format='7')


def plot_mesh_2d_bokeh(vertices, simplices, title=None, colored=False, default_axes=False):
    """Plot set of vertices and simplices (pandas DataFrames) interactively in 3D.
    """

    hv.extension('bokeh')

    if colored:
        edge_color = 'Z'
    else:
        edge_color = ''

    if default_axes:
        hooks = []
    else:
        hooks = [bokeh_axes_formatter]

    if title is None:
        title = ''

    # styling plot
    options = {'TriMesh': {'style': dict(cmap='viridis'
                                         ),
                           'plot': dict(width=900,
                                        height=450,
                                        aspect='square',
                                        show_grid=True,
                                        edge_color_index=edge_color,
                                        filled=colored,
                                        finalize_hooks=hooks,
                                        inspection_policy='edges',
                                        tools=['hover'],
                                        fontsize = {'title': '12pt'},
                                        toolbar = 'above'
                                        )}
               }

    trimesh = hv.TriMesh((simplices, vertices), label=title).opts(options)

    return trimesh


def plot_mesh_3d(vertices, simplices=None, title=None, step=1, colored=False, figsize=(16, 8), ax=None, viewpoint=(30, -60), **kwargs):
    """Plot point set (pandas DataFrame) in 3D triangular grid.
    """

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.elev = viewpoint[0]
        ax.azim = viewpoint[1]
    else:
        ax = ax

    if colored:
        colormap = 'viridis'
    else:
        colormap = None

    # Only accept step argument, if no simplices are specified (and therefore automatically Delaunay triangulated by matplotlib.
    if simplices is None:
        step = step
    else:
        step = 1

    ax.plot_trisurf(vertices.values[::step, 0], vertices.values[::step, 1], vertices.values[::step, 2],
                    triangles=simplices, cmap=colormap, **kwargs)
    if title is not None:
        ax.set_title(title, size='16')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')