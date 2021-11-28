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


# Module for trap analysis, based on graph theory

import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
from itertools import chain
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from copy import deepcopy
import sys


class Turtles(object):
    """Turtle class to find turtles on a surface. Stay tuned!

    Parameters:
        vertices (numpy.array[:,3]): Array of nodes.
        simplices (numoy.array[:,3]): Array of faces/triangles.
    """

    def __init__(self, vertices, simplices, keep_catchment=False, check_unique=True, xboundary='convex_hull', verbose=False):

        self._c = keep_catchment
        self._u = check_unique
        self._v = verbose
        self._xb = xboundary

        # keep vertices and simplices for reference
        self._vertices = vertices #numpy
        self._depth = self._vertices[:, 2] #numpy

        self._exterior_boundary = set()
        if self._xb == 'convex_hull':
            self._find_exterior_boundary_convex()
        elif self._xb == 'faces':
            self._find_exterior_boundary_faces(simplices)
        # TODO: Implement "no exterior boundary"-option!
        # elif self._xb == 'none':
        #     print('No exterior boundary!')
        else:
            print('Unknown boundary method, falling back to convex hull!')
            self._find_exterior_boundary_convex()


        # directed graph for exploration and undirected graph for operations on sets
        self._digraph = self._mesh_to_digraph(simplices) #networkx
        self._ungraph = self._digraph.to_undirected() #networkx

        # graph for hierarchical trap system
        self._system = nx.DiGraph()
        self._max_level = None

        # calculate global features
        self._local_maxima = []
        self._find_local_maxima()



        # calculate loops
        # m items
        self._initial_spill_points = []

        self._ca_dict = {}
        # n items (n <= m)
        self._pred_dict = {}
        self._trap_dict = {}
        # spill points now in system graph

        # start exploring!
        self._explore()

        # optional attributes to be filled by the user
        #self._volumes = []
        self._vol_dict = {}
        self._total_volume = None


    def _mesh_to_digraph(self, simplices):
        # input as numpy.array
        # output as networkx.DiGraph

        # TODO: Check vertices dataset for consistency
        # TODO: Maybe remove duplicate points in vertices (from marching cubes) --> But could lead to conflicts with simplices!

        if self._v:
            tqdm.write('Creating graph...')
        # extract edge pairs from simplices / triangular faces
        edges = np.concatenate((simplices[:, [0, 1]],
                                simplices[:, [1, 2]],
                                simplices[:, [0, 2]])
                               )

        edges_depth = np.stack((self._depth[edges[:, 0]], self._depth[edges[:, 1]]), axis=1)

        # arrange edge directions according to depth difference
        # for depth difference d(a) < d(b) the pairs will be (a,b)
        flip = edges_depth[:, 0] > edges_depth[:, 1]
        flipped_edges = np.where(np.stack((flip, flip), axis=1), np.fliplr(edges), edges)

        # identify equal edges, 'copy' them and flip them to add them to the directional ones
        # This way, we represent equal edges as 'loops': edges a-b and b-a.
        equal = edges_depth[:, 0] == edges_depth[:, 1]
        equal_edges = np.fliplr(edges[equal])

        # combine directed and loop edges, create graph
        directed_edges = np.concatenate((flipped_edges, equal_edges))
        graph = nx.DiGraph()
        graph.add_edges_from(directed_edges)

        # optional node attributes (not necessary for computations, but for drawing)
        # nx.set_node_attributes(graph, dict(enumerate(self.depth)), 'Z')

        return graph


    ### global functions ###

    def _find_exterior_boundary_convex(self):
        # The set of nodes that form the exterior boundary of the whole mesh in x,y direction

        ch = ConvexHull(self._vertices[:, 0:2], qhull_options='Qc')
        # combine the hull's spanning vertices and coplanar vertices that lie on the boundary
        self._exterior_boundary = set(np.concatenate((ch.vertices, ch.coplanar[:, 0])))

    def _find_exterior_boundary_faces(self, simplices):
        # extract edges from faces
        edges = np.concatenate((simplices[:, [0, 1]],
                                simplices[:, [1, 2]],
                                simplices[:, [0, 2]])
                               )

        # sort the edge pairs, so for an edge pair [a,b] always applies: a < b
        sorted_edges = np.sort(edges, axis=-1)

        # find edge pairs, that only occur once, so are only part of one face/triangle
        unique, counts = np.unique(sorted_edges, return_counts=True, axis=0)
        onetime = unique[counts == 1]

        # find unique nodes on boundary edges
        boundary_nodes = np.unique(onetime)

        self._exterior_boundary = set(boundary_nodes)

    def _find_local_maxima(self):
        # Returns a list of all maximum nodes, i.e. nodes that have no upward connection

        if self._v:
            tqdm.write('Determine local maxima...')
        self._local_maxima = [x for x in self._digraph.nodes() if self._digraph.out_degree(x) == 0]


    ### global loop functions ###

    def _explore(self):
        # main function

        # 1 - find initial catchment areas
        # temporary variable, might not be returned after execution of explore
        ca_dict = {}

        iter = tqdm(self._local_maxima, "Catch initial spill points") if self._v else self._local_maxima
        for i in iter:
            c = self.find_catchment_area(i) # set of all nodes
            p = self._find_spill_point(c) # int index of spill point
            ca_dict.update({i: c})
            self._initial_spill_points.append(p)

        if self._c is True:
            # Save catchment area globally only if desired
            self._ca_dict = ca_dict


        # 2 - find unique spill points and trap systems
        sp = self._initial_spill_points

        # init system
        edges = np.stack((sp, self._local_maxima)).T
        self._system.add_edges_from(edges)
        self._system.add_nodes_from(sp, level=1)
        # make sure local maxima get right level, even when lm == sp
        self._system.add_nodes_from(self._local_maxima, level=0)


        if self._u == True:
            current_level = 1
            while True:
                duplicates = [x for x in self._system.nodes() if (self._system.nodes[x]['level'] == current_level) & (self._system.out_degree(x) > 1)]

                if not duplicates:
                    break

                if self._v:
                    tqdm.write(str(len(duplicates)) + '\t duplicate spill points at level ' + str(current_level))

                for d in duplicates:
                    successors = self._system.successors(d)

                    # combine all catchment areas of duplicate spill points
                    new_ca = set()
                    for s in successors:
                        new_ca |= ca_dict[s]

                    new_sp = self._find_spill_point(new_ca)

                    # in special cases, a duplicate spill point does not define a next level trap
                    # which means that sp(duplicate) == duplicate
                    # this would cause a infinite loop due to increase of level!
                    if new_sp != d:
                        # add to graph
                        self._system.add_edge(new_sp, d)
                        self._system.node[new_sp]['level'] = current_level + 1

                        # add to dict
                        ca_dict.update({d: new_ca})

                #self.plot_hierarchy(figsize=(16,8), size=8) # DEBUG
                current_level += 1


        ### 3 - define traps only for unique spill points
        self._max_level = max(set(nx.get_node_attributes(self._system, 'level').values()))

        # iterate over each level
        iter = tqdm(range(self._max_level), "Define traps per level") if self._v else range(self._max_level)
        for level in iter:
            level_nodes = [x for x in self._system.nodes() if self._system.nodes[x]['level'] == level]

            # iterate over each node per level
            for node in level_nodes:

                # find spill_point
                predecessor = list(self._system.predecessors(node))

                # only applies to traps with spill point
                if predecessor:
                    trap_raw = set(self._find_trap(predecessor[0], ca_dict[node]))
                    self._pred_dict.update({node: predecessor[0]})

                    # find higher traps
                    successors = set(nx.bfs_tree(self._system, node)) - {node}

                    # only applies to nodes not on level 0
                    successor_traps = set()
                    for s in successors:
                        successor_traps |= self._trap_dict[s]

                    trap = trap_raw - successor_traps
                    self._trap_dict.update({node: trap})

        if self._c is True:
            # Save catchment area globally only if desired
            self._ca_dict = ca_dict


    # def _find_non_unique(self, spill_points):
    #     unique, counts = np.unique(spill_points, return_counts=True)
    #     non_unique = unique[counts > 1]
    #     return non_unique

    ### single feature functions ###

    def find_catchment_area(self, local_max):
        sub_graph = nx.bfs_tree(self._digraph, local_max, reverse=True).reverse()
        catchment_area = set(sub_graph.nodes())
        return catchment_area


    def find_interior_boundary(self, catchment_area):
        # complementary set of area in graph
        graph = set(self._ungraph.nodes)
        comp = graph - catchment_area
        ibdy = set(chain.from_iterable(self._ungraph[v] for v in comp)) - comp
        return ibdy


    def _find_spill_point(self, catchment_area):
        # TRANSITION: catchment area of local maximum
        # catchment_area = self.find_catchment_area(node)

        # union of interior catchment boundary and exterior mesh boundary
        boundary = self.find_interior_boundary(catchment_area) | self._exterior_boundary
        # intersect to find boundary of catchment area
        spill_candidates = catchment_area & boundary

        # highest point on boundary --> spill point
        spill_candidates = list(spill_candidates)
        spill_point = spill_candidates[self._depth[spill_candidates].argmax()]

        return spill_point

    def _find_trap(self, spill_point, catchment_area):
        depth_mask = np.where(self._depth >= self._depth[spill_point])
        trap = np.intersect1d(list(catchment_area), depth_mask)

        return trap

    def _count_dict_values(self, dictionary):
        # counts dict values and returns a new dict with same keys and count
        count = {}
        for key, value in dictionary.items():
            count.update({key: len(value)})
        return count

    #def find_trap_set(self, spill_point, catchment_area):
    #    depth_mask = set(np.where(self.depth >= self.depth[spill_point])[0])
    #    trap = catchment_area & depth_mask
    #    return list(trap)


    ### Non-mandatory functions ###

    def calculate_volumes(self):
        self._vol_dict = {}

        iter = tqdm(self._trap_dict.items(), 'Calculate volumes') if self._v else self._trap_dict.items()
        for i, trap in iter:
            if len(list(trap)) >= 3:
                try:
                    vol = self._calculate_volume(list(trap))
                except:
                    if self._v:
                        tqdm.write('*** Problema at trap ' + str(i) + ':' + str(sys.exc_info()[1]))
                    vol = np.nan
            else:
                vol = np.nan

            self._vol_dict.update({i: vol})

        self._total_volume = np.nansum(list(self._vol_dict.values()))

        # self._volumes = []
        #
        # iter = tqdm(self._traps, "Calculate volumes") if self._v else self._traps
        # for i, trap in enumerate(iter):
        #     if len(trap) >= 3:
        #         try:
        #             self._volumes.append(self._calculate_volume(trap))
        #         except:
        #             if self._v:
        #                 tqdm.write('*** Problema at trap ' + str(i) + ':' + str(sys.exc_info()[1]))
        #             self._volumes.append(np.nan)
        #     else:
        #         self._volumes.append(np.nan)

    def _calculate_volume(self, trap_nodes):
        top = self._vertices[trap_nodes]
        bottom = top.copy()
        bottom[:, 2] = top[:, 2].min()
        convex_hull = ConvexHull(np.concatenate((top, bottom)))

        return convex_hull.volume

    def summary(self):
        # requires pandas 0.24 for using NaN values in integer series (via 'Int64')
        volume = pd.Series(self._vol_dict)
        size = pd.Series(self._count_dict_values(self._trap_dict), dtype='Int64')
        level = pd.Series(nx.get_node_attributes(self._system, 'level'))
        predecessor = pd.Series(self._pred_dict, dtype='Int64').sort_values()

        summary = pd.DataFrame({'level': level, 'spill point': predecessor, 'trap size': size, 'volume': volume})
        summary.index.name = 'top point'
        summary.reset_index(inplace=True, drop=False)
        summary = summary[['level', 'top point', 'spill point', 'trap size', 'volume']]
        # TODO: Add cumulative volume?

        return summary

    def plot_volumes(self, color='#542E71', figsize=(16,8)):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 5))

        summary = self.summary()
        df = summary[['top point', 'volume']].dropna().sort_values(by='volume', ascending=False)

        fig, ax = plt.subplots(figsize=figsize)
        df.plot.bar(x='top point', y='volume', color=color, ax=ax)
        ax.set_xlabel('Trap top point')
        ax.set_ylabel('Volume')
        ax.yaxis.set_major_formatter(formatter)
        ax.get_legend().remove()
        #plt.show()

    def plot_level_volume(self, palette='tab20', figsize=(16,8)):

        summary = self.summary()
        df = summary.groupby('level')[['volume']].sum().reset_index()

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(ax=ax, x='level', y='volume', palette=palette, data=df)
        ax.set_xlabel('Trap level')
        ax.set_ylabel('Cumulative volume')
        #plt.show()

    def plot_hierarchy(self, root=None, palette='tab20', size=15, figsize=(16,8)):

        levels = list(nx.get_node_attributes(self._system, 'level').values())
        n_levels = len(set(levels))
        pos = nx.nx_pydot.pydot_layout(self._system, prog='dot', root=root)
        flipped_pos = {node: (x, -y) for (node, (x, y)) in pos.items()}

        # hack to 'prevent' normalization of qualitative colormaps
        # all seaborn palettes can be passed
        import matplotlib as mpl
        import seaborn as sns
        colors = sns.color_palette(palette=palette, n_colors=100)
        cmap = mpl.colors.ListedColormap(colors, name='from_list', N=n_levels)

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw(self._system, pos=flipped_pos, ax=ax,
                with_labels=True, node_size=size * 150, node_color=levels, cmap=cmap,
                arrows=True, edge_color='black', linewidths=1, font_size=size)
        #plt.show()

    ### Getters

    @property
    def local_maxima(self):
        return self._local_maxima

    @property
    def exterior_boundary(self):
        # transform from set to list
        return list(self._exterior_boundary)

    @property
    def catchment_areas(self):
        if not self._ca_dict:
            print('No catchment areas stored in object!')
            print('Use option ´keep_catchment´ or the function ´find_catchment_area(local_max)´.')
        else:
            # transform from nested sets to nested lists
            catchment_list = []
            for i in self._ca_dict.values():
                catchment_list.append(list(i))
            return catchment_list

    @property
    def spill_points(self):
        return list(set(self._pred_dict.values()))
        # TODO: Maybe rename predecessors in general?

    @property
    def traps(self):
        # transform from nested numpys to nested lists
        trap_list = []
        for i in self._trap_dict.values():
            trap_list.append(list(i))
        return trap_list

    @property
    def volumes(self):
        if not self._vol_dict:
            self.calculate_volumes()
        return list(self._vol_dict.values())

    @property
    def total_volume(self):
        if not self._vol_dict:
            self.calculate_volumes()
        return self._total_volume
