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
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

# vtk
from vtk import (vtkPolyData, vtkPoints, vtkTriangle, vtkCellArray,
                 vtkDecimatePro, vtkQuadricDecimation, vtkQuadricClustering,
                 vtkPolyDataPointSampler,
                 vtkDataSetWriter, vtkPolyDataWriter, vtkIVWriter, vtkPLYWriter)
from vtk.util import numpy_support


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


def to_vtk(vertices, simplices=None):
    '''
    Creates a VTK mesh (vtkPolyData object) from a given point set utilizing Delaunay triangulation in the first place.
    
    Arguments:
        vertices: pandas DataFrame of xyz coordinates
        simplices: pandas DataFrame of simplices from Delaunay triangulation, if None basic triangulation will be performed

    Returns:
        polydata: vtkPolyData object which serves as input for subsequent vtk simplification / decimation operations   
    '''

    if simplices is None:
        ## Delaunay
        # triangulation = Delaunay(vertices.iloc[:,:2].values)
        # simplices = triangulation.simplices
        simplices = triangulate(vertices)
    else:
        simplices = simplices

    ## VTK polygons

    # initiate points and read vertices
    points = vtkPoints()

    for p in vertices.values:
        points.InsertNextPoint(p)

    # initiate triangles and read simplices        
    # Unfortunately in this simple example the following lines are ambiguous.
    # The first 0 is the index of the triangle vertex which is ALWAYS 0-2.
    # The second 0 is the index into the point (geometry) array, so this can range from 0-(NumPoints-1)
    # i.e. a more general statement is triangle->GetPointIds()->SetId(0, PointId);
    triangle = vtkTriangle()
    triangles = vtkCellArray()

    for i in simplices.values:
        triangle.GetPointIds().SetId(0, i[0])
        triangle.GetPointIds().SetId(1, i[1])
        triangle.GetPointIds().SetId(2, i[2])
        triangles.InsertNextCell(triangle)

    # combine in PolyData object      
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    # check
    polydata.Modified()

    return polydata


def to_pandas(inputdata):
    '''
    Extract two sets (pandas DataFrame) of vertices and simplices from a given VTK mesh (vtkPolyData object).

    Arguments:
        inputdata: vtkPolyData object which serves as input for subsequent vtk simplification / decimation operations

    Returns:
        vertices: pandas DataFrame of xyz coordinates
        simplices: pandas DataFrame of abc simplices
    '''

    # copy to prevent recursive updating of values
    polydata = vtkPolyData()
    polydata.DeepCopy(inputdata)

    # extract vertices and simplices
    vertices = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())

    triangles = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    # reshape array and extract only the last 3 of 4 values per simplex
    simplices = triangles.reshape(-1, 4)[:, 1:4]

    # convert into pandas
    vertices = pd.DataFrame(data=vertices, columns=['X', 'Y', 'Z'])
    simplices = pd.DataFrame(data=simplices, columns=['A', 'B', 'C'])

    return vertices, simplices


# def point_it (polydata):
#
#     '''
#     Extract a point set (pandas DataFrame) / the vertices from a given VTK mesh (vtkPolyData object).
#
#     Arguments:
#         polydata: vtkPolyData object which serves as input for subsequent vtk simplification / decimation operations
#
#     Returns:
#         points: pandas DataFrame of xyz coordinates
#     '''
#
#     points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
#
#     points = pd.DataFrame(data=points, columns=['X', 'Y', 'Z'])
#
#     return points


def print_decimation_result(title_string, inputPoly, decimatedPoly, parameter_string=None):
    print(
        str(title_string) + ', parameters: ' + str(parameter_string) + '\n'
                                                                       'Points:\t\t' + str(
            inputPoly.GetNumberOfPoints()) + '\t-->\t' + str(decimatedPoly.GetNumberOfPoints()) + '\n'
                                                                                                  'Polygons:\t' + str(
            inputPoly.GetNumberOfPolys()) + '\t-->\t' + str(decimatedPoly.GetNumberOfPolys()) + '\n'
    )


def hausdorff_uni(a, b, seed=None):
    return directed_hausdorff(a.values, b.values, seed=seed)[0]


def hausdorff_bi(a, b, seed=None):
    return np.max(directed_hausdorff(a.values, b.values, seed=seed)[0],
                  directed_hausdorff(b.values, a.values, seed=seed)[0])


def vtk_decimate_pro(inputdata,
                     target_reduction=.9,
                     preserve_topology=False,
                     feature_angle=15.0,
                     splitting=True,
                     split_angle=75.0,
                     pre_split_mesh=False,
                     maximum_error=1.0e+299,
                     accumulate_error=False,
                     boundary_vertex_deletion=True,
                     degree=25,
                     inflection_point_ratio=10.0,
                     verbose=False
                     ):
    """Wrapper for VTK DecimatePro

    Modified after Schroeder et al. (1992)
    See documentation: https://www.vtk.org/doc/nightly/html/classvtkDecimatePro.html#details

    Arguments:
        inputdata (vtk.vtkPolyData): ytk object containing vertices and simplices
        target_reduction (double): Desired reduction in the total number of polygons (default: .9).
        preserve_topology (bool): Turn on/off whether to preserve the topology of the original mesh (default: False).
        feature_angle (double): Specify the mesh feature angle (default 15.0).
        splitting (bool): Turn on/off the splitting of the mesh at corners, along edges, at non-manifold points, or anywhere else a split is required (default True).
        split_angle (double):  Specify the mesh split angle (default 75.0).
        pre_split_mesh (bool): In some cases you may wish to split the mesh prior to algorithm execution (default False).
        maximum_error (double): Largest decimation error that is allowed during the decimation process (default 1.0e+299).
        accumulate_error (bool): The computed error can either be computed directly from the mesh or the error may be accumulated as the mesh is modified (default False).
        boundary_vertex_deletion (bool): Deletion of vertices on the boundary of a mesh (default True).
        degree (int): If the number of triangles connected to a vertex exceeds "Degree", then the vertex will be split (default 25).
        inflection_point_ratio (double): Specify the inflection point ratio (default 10.0)
        verbose (bool): Print out steps and basic statistics (default False).

    Returns:
        decimatedPoly: vtkPolyData object
    """

    title_string = 'VTK DecimatePro'

    if not isinstance(inputdata, vtkPolyData):
        raise TypeError('Unknown dtype for inputdata! vtk.vtkPolyData expected.')

    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(inputdata)

    decimate = vtkDecimatePro()
    decimate.SetInputData(inputPoly)

    # set parameters
    decimate.SetTargetReduction(target_reduction)
    decimate.SetPreserveTopology(preserve_topology)
    decimate.SetFeatureAngle(feature_angle)
    decimate.SetSplitting(splitting)
    decimate.SetSplitAngle(split_angle)
    decimate.SetPreSplitMesh(pre_split_mesh)
    decimate.SetMaximumError(maximum_error)
    decimate.SetAccumulateError(accumulate_error)
    decimate.SetBoundaryVertexDeletion(boundary_vertex_deletion)
    decimate.SetDegree(degree)
    decimate.SetInflectionPointRatio(inflection_point_ratio)

    # execute
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print_decimation_result(title_string, inputPoly, decimatedPoly, ('target_reduction=' + str(target_reduction)))

    return decimatedPoly


def vtk_quadric_decimation(inputdata,
                           target_reduction=.9,
                           volume_preservation=False,
                           attribute_error_metric=False,
                           scalars_weight=.1,
                           vectors_weight=.1,
                           normals_weight=.1,
                           tcoords_weight=.1,
                           tensors_weight=.1,
                           verbose=False):
    """Wrapper for VTK QuadricDecimation

    Based on Garland and Heckbert (1997)
    See documentation: https://www.vtk.org/doc/nightly/html/classvtkQuadricDecimation.html#details

    Arguments:
        inputdata (vtk.vtkPolyData): vtk object containing vertices and simplices
        target_reduction (double): Desired reduction (expressed as a fraction of the original number of triangles) (default: .9).
        volume_preservation (bool): Decide whether to activate volume preservation which greatly reduces errors in triangle normal direction (default: False).
        attribute_error_metric (bool): Decide whether to include data attributes in the error metric. If False, the attribute weights will be ignored (default False).

        scalars_weight (double): (default .1).
        vectors_weight (double): (default .1).
        normals_weight (double): (default .1).
        tcoords_weight (double): (default .1).
        tensors_weight (double): (default .1).

        If a weight is set to None, it will be deactivated.

        verbose (bool): Print out steps and basic statistics (default False).

    Returns:
        decimatedPoly: vtkPolyData object
    """

    title_string = 'VTK Quadric Decimation'

    if not isinstance(inputdata, vtkPolyData):
        raise TypeError('Unknown dtype for inputdata! vtk.vtkPolyData expected.')

    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(inputdata)

    decimate = vtkQuadricDecimation()
    decimate.SetInputData(inputPoly)

    # set parameters
    decimate.SetTargetReduction(target_reduction)
    decimate.SetVolumePreservation(volume_preservation)
    decimate.SetAttributeErrorMetric(attribute_error_metric)

    if scalars_weight != None:
        decimate.SetScalarsAttribute(True)
        decimate.SetScalarsWeight(scalars_weight)
    else:
        decimate.SetScalarsAttribute(False)

    if vectors_weight != None:
        decimate.SetVectorsAttribute(True)
        decimate.SetVectorsWeight(vectors_weight)
    else:
        decimate.SetVectorsAttribute(False)

    if normals_weight != None:
        decimate.SetNormalsAttribute(True)
        decimate.SetNormalsWeight(normals_weight)
    else:
        decimate.SetNormalsAttribute(False)

    if tcoords_weight != None:
        decimate.SetTCoordsAttribute(True)
        decimate.SetTCoordsWeight(tcoords_weight)
    else:
        decimate.SetTCoordsAttribute(False)

    if tensors_weight != None:
        decimate.SetTensorsAttribute(True)
        decimate.SetTensorsWeight(tensors_weight)
    else:
        decimate.SetTensorsAttribute(False)

    # execute
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print_decimation_result(title_string, inputPoly, decimatedPoly, ('target_reduction=' + str(target_reduction)))

    return decimatedPoly


def vtk_quadric_clustering(inputdata,
                           divisions=[50, 50, 50],
                           division_spacing=None,
                           auto_adjust_divisions=True,
                           use_input_points=False,
                           verbose=False):
    """Wrapper for VTK QuadricClustering

    Based on Lindstrom (2000)
    See documentation: https://www.vtk.org/doc/nightly/html/classvtkQuadricClustering.html#details

    Arguments:
        inputdata (vtk.vtkPolyData): ytk object containing vertices and simplices
        divisions ([int,int,int]): number of subdivisions in x,y and z direction (default [50, 50, 50]).
        division_spacing([double, double, double]): This is an alternative way to set up the bins. If you are trying to match boundaries between pieces, then you should use these methods rather than SetNumberOfDivisions. To use these methods, specify the origin and spacing of the spatial binning (default None instead of [1.,1.,1.]).
        auto_adjust_divisions (bool): Enable automatic adjustment of number of divisions. If off, the number of divisions specified by the user is always used (as long as it is valid) (default True).
        use_input_points (bool): Normally the point that minimizes the quadric error function is used as the output of the bin. When this flag is on, the bin point is forced to be one of the points from the input (the one with the smallest error) (default False).
        verbose (bool): Print out steps and basic statistics (default False).

    Returns:
        decimatedPoly: vtkPolyData object
    """

    title_string = 'VTK Quadric Clustering'

    if not isinstance(inputdata, vtkPolyData):
        raise TypeError('Unknown dtype for inputdata! vtk.vtkPolyData expected.')

    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(inputdata)

    decimate = vtkQuadricClustering()
    decimate.SetInputData(inputPoly)

    # set parameters
    # apparently, AutoAdjustNumberOfDivisions does not affect DivisionSpacing
    if division_spacing is None:
        decimate.SetNumberOfDivisions(divisions)
        decimate.SetAutoAdjustNumberOfDivisions(auto_adjust_divisions)
    else:
        decimate.SetDivisionSpacing(division_spacing)

    decimate.SetUseInputPoints(use_input_points)

    # execute
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    if verbose:
        print_decimation_result(title_string, inputPoly, decimatedPoly,
                                ('(auto-adjusted) divisions:' + str(decimate.GetNumberOfDivisions())))

    return decimatedPoly


def random_decimation(input_data,
                      target_reduction=.9,
                      seed=None,
                      verbose=False
                      ):
    """Random Decimation

    Arguments:
        inputdata (vtk.vtkPolyData): ytk object containing vertices and simplices
        target_reduction (double): Desired reduction (expressed as a fraction of the original number of points/vertices) (default .9).
        seed (int): Optional seed (default None)
        verbose (bool): Print out steps and basic statistics (default False).

    Returns:
        decimatedPoly: vtkPolyData object
    """

    title_string = 'Random Decimation'

    if isinstance(input_data, vtkPolyData):
        input_ver, input_sim = to_pandas(input_data)
        input_poly = input_data
    else:
        raise TypeError('Unknown dtype for inputdata! vtk.vtkPolyData expected.')

    input_size = input_ver.shape[0]
    decimated_size = round(input_size * (1 - target_reduction))

    if seed is not None:
        np.random.seed(seed)

    decimated_index = np.random.randint(0, input_size, decimated_size, )
    decimated_ver = input_ver.iloc[decimated_index]

    if verbose:
        # transform into decimated mesh to report number of polygons
        decimated_poly = to_vtk(decimated_ver)
        print_decimation_result(title_string, input_poly, decimated_poly, ('target_reduction=' + str(target_reduction)))

    return to_vtk(decimated_ver)


def vtk_mesh_sampling(inputdata, distance=150, generate_vertex_points=True, generate_edge_points=True,
                      generate_face_points=True, verbose=False):
    """Sample points on a mesh from vertices, edges and faces in a given distance

      Arguments:
          inputdata (vtk.vtkPolyData): ytk object containing vertices and simplices
          distance (float): Set the approximate distance between points. This is an absolute distance measure (default here 150 for safety reasons).
          generate_vertex_points (bool): Indicating whether cell vertex points should be output (default True).
          generate_edge_points (bool): Indicating whether cell edges should be sampled to produce output points (default True).
          generate_face_points (bool): Indicating whether cell interiors should be sampled to produce output points (default True).

          verbose (bool): Print out basic statistics (default False).

      Returns:
          decimatedPoly: vtkPolyData object or pandas.DataFrame of xyz coordinates (depends on input)
      """

    if not isinstance(inputdata, vtkPolyData):
        raise TypeError('Unknown dtype for inputdata! vtk.vtkPolyData expected.')

    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(inputdata)

    sampler = vtkPolyDataPointSampler()
    sampler.SetInputData(inputPoly)

    # set parameters
    sampler.SetDistance(distance)
    sampler.SetGenerateVertexPoints(generate_vertex_points)
    sampler.SetGenerateEdgePoints(generate_edge_points)
    sampler.SetGenerateInteriorPoints(generate_face_points)

    sampler.Update()

    sampler_poly = vtkPolyData()
    sampler_poly.ShallowCopy(sampler.GetOutput())

    if verbose:
        print('Points sampled:', sampler_poly.GetNumberOfPoints())

    return sampler_poly


def evaluate_legacy(poly_list, method_description, original_vertices, targets=None, **kwargs):
    '''Evaluate a given list of simplified vtkPolyData objects against
    original data points/vertices by calculating directed and bidirectional Hausdorff distances
    '''

    cols = ['description', 'target',
            'size', 'hausdorff_a', 'hausdorff_b', 'hausdorff',
            'sample_size', 'sample_hausdorff_a', 'sample_hausdorff_b', 'sample_hausdorff']

    df = pd.DataFrame(columns=cols, index=range(len(poly_list)))

    for i, poly in enumerate(poly_list):
        df.iloc[i]['description'] = method_description

        if targets is not None:
            df.iloc[i]['target'] = targets[i]

        df.iloc[i]['size'] = poly.GetNumberOfPoints()
        df.iloc[i]['hausdorff_a'] = hausdorff_uni(original_vertices, to_pandas(poly)[0])
        df.iloc[i]['hausdorff_b'] = hausdorff_uni(to_pandas(poly)[0], original_vertices)

        poly_sampled = vtk_mesh_sampling(poly, **kwargs)
        df.iloc[i]['sample_size'] = poly_sampled.GetNumberOfPoints()
        df.iloc[i]['sample_hausdorff_a'] = hausdorff_uni(original_vertices, to_pandas(poly_sampled)[0])
        df.iloc[i]['sample_hausdorff_b'] = hausdorff_uni(to_pandas(poly_sampled)[0], original_vertices)

    df.loc[:, 'hausdorff'] = df[['hausdorff_a', 'hausdorff_b']].max(axis=1)
    df.loc[:, 'sample_hausdorff'] = df[['sample_hausdorff_a', 'sample_hausdorff_b']].max(axis=1)

    return df

def evaluate(poly_list, method_description, original_poly, targets=None, sample_original=False, verbose=False, **kwargs):
    '''Evaluate a given list of simplified vtkPolyData objects against
    original data points/vertices by calculating directed and bidirectional Hausdorff distances
    '''

    cols = ['description', 'target', 'size',
            'original_samples', 'reduced_samples',
            'hausdorff_a', 'hausdorff_b', 'hausdorff']

    df = pd.DataFrame(columns=cols, index=range(len(poly_list)))

    original_sampled_vtk = vtk_mesh_sampling(original_poly, **kwargs)
    original_sampled_n = original_sampled_vtk.GetNumberOfPoints()
    original_sampled = to_pandas(original_sampled_vtk)[0]

    original = to_pandas(original_poly)[0]
    original_n = original_poly.GetNumberOfPoints()

    iter = tqdm(poly_list, "Calculate Hausdorff distances") if verbose else poly_list
    for i, poly in enumerate(iter):

        df.iloc[i]['description'] = method_description

        if targets is not None:
            df.iloc[i]['target'] = targets[i]

        df.iloc[i]['size'] = poly.GetNumberOfPoints()

        poly_sampled_vtk = vtk_mesh_sampling(poly, **kwargs)
        poly_sampled = to_pandas(poly_sampled_vtk)[0]
        df.iloc[i]['reduced_samples'] = poly_sampled_vtk.GetNumberOfPoints()

        if sample_original:
            df.iloc[i]['original_samples'] = original_sampled_n
            df.iloc[i]['hausdorff_a'] = hausdorff_uni(original_sampled, poly_sampled)
            df.iloc[i]['hausdorff_b'] = hausdorff_uni(poly_sampled, original_sampled)
        else:
            df.iloc[i]['original_samples'] = original_n
            df.iloc[i]['hausdorff_a'] = hausdorff_uni(original, poly_sampled)
            df.iloc[i]['hausdorff_b'] = hausdorff_uni(poly_sampled, original)

    df.loc[:, 'hausdorff'] = df[['hausdorff_a', 'hausdorff_b']].max(axis=1)

    return df


def vtk_export(polydata, filename, method='vtk'):
    """Export VTK polyData object to different file formats.

    polydata (vtk.vtkPolyData): ytk object containing vertices and simplices
    filename (string): path and filename without extension
    method (string): one of the following options

    vtk: VTK legacy file version 4.2 (e.g. for import into ParaView)
    ply: Stanford University ".ply" file (e.g. for import into MeshLab)
    iv: OpenInventor 2.0 file (e.g. for import into METRO)
    """

    # TODO: Binary ASCII stuff

    if method == 'vtk':
        export = vtkPolyDataWriter()
        export.SetInputDataObject(polydata)
        export.SetFileName(filename + '.vtk')
        export.Write()
    elif method == 'ply':
        export = vtkPLYWriter()
        export.SetInputDataObject(polydata)
        export.SetFileName(filename + '.ply')
        export.Write()
    elif method == 'iv':
        export = vtkIVWriter()
        export.SetInputDataObject(polydata)
        export.SetFileName(filename + '.iv')
        export.Write()
    else:
        print('File format unknown!')