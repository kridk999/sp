import numpy as np
import os
from collections import defaultdict
import glob
import SimpleITK as sitk
from skimage.measure import label
from skimage.morphology import ball, binary_closing
from sklearn.mixture import GaussianMixture
from scipy.ndimage import center_of_mass
from scipy.interpolate import interpn, make_interp_spline
import pandas as pd
from PIL import Image
import tqdm
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import vtk 
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from matplotlib import cm  # For colormap
from matplotlib.colors import Normalize

import imageio


def spline_interp(y, x=None, n_points=1000):
    if x is None:
        x = np.arange(len(y))
    spl = make_interp_spline(x, y, k=3)
    
    x_new = np.linspace(x[0], x[-1], n_points)
    y_smooth = spl(x_new)
    return x_new, y_smooth

def plot_bloodpool(path, split):
    
    df = pd.read_csv(os.path.join(path,split["patient_id"].iloc[0].strip(),"LV_blood_stats.csv"))
    vols = []
    for ind in split.index:
        patient = split["pseudonymized_id"][ind]
        val = df[df.id_seg==patient]["tot_vol_mm3"]
        vols.append(val.item())

    all_vols = np.array(vols)
    ef = (max(vols) - min(vols))/max(vols)

    spl = make_interp_spline(np.arange(0,len(all_vols)), all_vols, k=3)
    x_new = np.linspace(0, len(all_vols)-1, 300)
    vol_smooth = spl(x_new)
    plot_path = os.path.join(path, "HU_plots")
    
    os.makedirs(plot_path, exist_ok=True)

    # plot ejection stuff
    plt.figure(figsize=(6,5))
    plt.plot(x_new, vol_smooth)

    plt.xlabel("Scan percentage")
    plt.xticks(np.arange(0,len(vols),2))

    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=20.0))
    plt.ylabel("Blood volume (mm3)")
    plt.title(f"LV blood over CFA, EF: {ef:.2f}")
    plt.show()


def extract_largest_CC(lab):
    labels = label(lab)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = largestCC.astype(np.int16)
    return largestCC

def extract_n_largest_CC(lab, n):
    labels = label(lab)
    assert( labels.max() != 0 ) # assume at least 1 CC
    bincount = np.bincount(labels.flat)[1:]
    if len(bincount) < n:
        print(f"warning: only {len(bincount)} CCs found, returning all")
        n = len(bincount)
    largestCC = np.zeros_like(lab)
    for i in range(n):
        largestCC[labels == np.argmax(bincount)+1] = i
        bincount[np.argmax(bincount)] = 0
    largestCC = largestCC.astype(np.int16)
    return largestCC

def calculate_mean_std(im, lab_all, radii_pct=0.02):
    lab_LV = (lab_all==1).astype(np.int16)
    com = center_of_mass(lab_LV)
    radius = radii_pct * min(lab_LV.shape)
    sphere = ball(radius)
    sphere_nnz = sphere.nonzero()
    grid = (np.arange(0,lab_LV.shape[0]),
            np.arange(0,lab_LV.shape[1]),
            np.arange(0,lab_LV.shape[2]))
    sample_points = np.array([sphere_nnz[0]+com[0]-sphere.shape[0]/2,
                              sphere_nnz[1]+com[1]-sphere.shape[1]/2,
                              sphere_nnz[2]+com[2]-sphere.shape[2]/2]).T
    HU_vals = interpn(grid, im, sample_points)
    # print("\n no. HU", len(HU_vals))
    return HU_vals.mean(), HU_vals.std()

def get_com(label, as_int=False, real_coords=False):
    """
        Get centre of mass of a SimpleITK.Image

    Args:
        label (sitk.Image): Label mask image.
        as_int (bool, optional): Returns each components as int if true. Defaults to True.
        real_coords (bool, optional): Return coordinates in physical space if true. Defaults to
            False.

    Returns:
        list: List of coordinates
    """
    if isinstance(label, np.ndarray):
        com = center_of_mass(label)

    else:
        arr = sitk.GetArrayFromImage(label)
        com = center_of_mass(arr)[::-1]

        if real_coords:
            com = label.TransformContinuousIndexToPhysicalPoint(com)

    return com

def calculateSphere(shape, center, radius_pct = 0.02):
    Z, X, Y = np.meshgrid(np.arange(-shape[0]//2,shape[0]//2),
                            np.arange(-shape[1]//2,shape[1]//2),
                            np.arange(-shape[2]//2,shape[2]//2))
    
    radiusX = radius_pct * shape[1]
    radiusY = radius_pct * shape[2]
    radiusZ = radius_pct * shape[0]
    
    sphere = (np.power((X)/radiusX,2)
            +np.power((Y)/radiusY,2)
            +np.power((Z)/radiusZ,2)<=1)
    sphere_nonzero = sphere.nonzero()
    return sphere_nonzero


def plot_gmm(im, lab):
    
    values = im[lab.astype(bool)].reshape(-1,1)
    
    gmm = GaussianMixture(n_components=5,
                        covariance_type="full",
                        #   means_init=np.array([[100,300,600]]).T
                        ).fit(values)
    
    pred = gmm.predict(values)+1
    print(gmm.means_)
    x = np.linspace(min(values), max(values), 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.figure()
    plt.hist(values, 500, density=True, histtype='stepfilled', alpha=0.4)
    plt.plot(x, pdf, '-k')
    plt.plot(x, pdf_individual, '--k')
    plt.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top')
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    plt.show()


def segmentation_vtk_to_mesh_vtk(path, sigma=1.0, save_name=None, label_id=1):

    reader_image = vtk.vtkNIFTIImageReader()
    reader_image.SetFileName(path)
    reader_image.Update()

    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(reader_image.GetOutput())
    surface.SetNumberOfContours(1)
    surface.SetValue(0, label_id)
    surface.Update()
    surface.ComputeNormalsOn()
    surface.ComputeGradientsOn()
    surface.Update()

    connectivity_filter = vtk.vtkConnectivityFilter()
    connectivity_filter.SetInputConnection(surface.GetOutputPort())
    connectivity_filter.SetExtractionModeToLargestRegion()
    connectivity_filter.Update()   
   
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(connectivity_filter.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(15.0)
    smoother.SetPassBand(0.1)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update() 

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(15.0)
    normals.Update()
    
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(normals.GetOutputPort())
    cleaner.ToleranceIsAbsoluteOn()
    cleaner.SetAbsoluteTolerance(0.001)
    cleaner.Update() 
    # fill_filter = vtk.vtkFillHolesFilter()
    # fill_filter.SetInputData(cleaner.GetOutput())
    # fill_filter.SetHoleSize(100.0) # Set the maximum hole size to fill
    # fill_filter.Update()

    mesh = cleaner.GetOutput()
    print(f"Number of points in mesh: {mesh.GetNumberOfPoints()}")
    if save_name:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(mesh)
        
        writer.SetFileName(save_name)
        writer.Write()
    return mesh

def read_vtk_mesh(path, get_normals=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    output = reader.GetOutput()
    assert isinstance(output, vtk.vtkPolyData), f"Output is not a vtkPolyData object: {path}"
    assert output.GetNumberOfPoints() > 0, f"No points found in mesh: {path}"

    if get_normals and not output.GetPointData().GetNormals():
        # vtkTriangleMeshPointNormals 
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(output)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.Update()
        output = normals.GetOutput()

    return output

def write_vtk_mesh(mesh, path):
    # Remove the existing file if it exists
    if os.path.isfile(path):
        os.remove(path)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileTypeToASCII()
    writer.SetFileVersion(42)
    writer.SetFileName(path)
    writer.Write()

    # Check if the file was written
    assert os.path.isfile(path), f"Mesh not written to file: {path}"

def read_vtk_volume(path, return_reader=False):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(path)
    reader.Update()
    output = reader.GetOutput()
    assert isinstance(output, vtk.vtkImageData), "Output is not a vtkImageData object"
    return output if not return_reader else reader

def read_vtk_and_get_sampler(path,use_nn=False):
    volume = read_vtk_volume(path)
    reslicer = vtk.vtkImageReslice()
        
    if use_nn:
        castFilter = vtk.vtkImageCast()
        castFilter.SetOutputScalarTypeToUnsignedChar()  # or another appropriate type
        castFilter.SetInputData(volume)
        castFilter.Update()

        image = castFilter.GetOutput()
        reslicer.SetInputData(image)
        reslicer.Update()
        reslicer.GetOutput()
        sampler = vtk.vtkImageReslice()
        sampler.SetInputData(reslicer.GetOutput())
        sampler.SetInterpolationModeToNearestNeighbor()
    else:
        reslicer.SetInputData(volume)
        reslicer.Update()
        reslicer.GetOutput()
        sampler = vtk.vtkImageReslice()
        sampler.SetInputData(reslicer.GetOutput())
        sampler.SetInterpolationModeToLinear()
    sampler.AutoCropOutputOn()
    sampler.Update()
    return sampler.GetOutput()

def polar2cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def mesh_to_vol(mesh, path, save_name=None, return_vtk_format=False):
    
    total_path = os.path.join(path, "segmentations","total_seg", "total_seg.nii.gz")

    reader_total = vtk.vtkNIFTIImageReader()
    reader_total.SetFileName(total_path)
    reader_total.Update()

    volume = vtk.vtkImageData()
    volume.SetDimensions(reader_total.GetOutput().GetDimensions())  # Adjust the dimensions as needed
    volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Create a stencil from the mesh
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(reader_total.GetOutput().GetOrigin())
    stencil.SetOutputSpacing(reader_total.GetOutput().GetSpacing())
    stencil.SetOutputWholeExtent(volume.GetExtent())
    stencil.Update()

    # Apply the stencil to the volume
    imgstencil = vtk.vtkImageStencil()
    imgstencil.SetInputData(volume)
    imgstencil.SetStencilData(stencil.GetOutput())
    imgstencil.ReverseStencilOn()
    imgstencil.SetBackgroundValue(1)
    imgstencil.Update()

    volume = imgstencil.GetOutput()


    volume_array = vtk.util.numpy_support.vtk_to_numpy(volume.GetPointData().GetScalars())
    volume_array = np.reshape(volume_array, reader_total.GetOutput().GetDimensions()[::-1])  # Adjust the shape as per your volume size
    if return_vtk_format:
        return volume_array
    volume_array = volume_array[::-1,:,:]

    # label_total = sitk.ReadImage(total_path)
    # lab_total = sitk.GetArrayFromImage(label_total).astype(np.uint8)

    label_total = sitk.ReadImage(total_path)

    vol = sitk.GetImageFromArray(volume_array.astype(np.uint8))
    vol.CopyInformation(label_total)
    if save_name:
        sitk.WriteImage(vol, os.path.join(path, save_name))
    return vol

def points_to_mesh(points, save_name=None):
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(points)
    delaunay.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(delaunay.GetOutput())
    geometry_filter.Update()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(geometry_filter.GetOutput())
    fill_holes.SetHoleSize(1.0)
    fill_holes.Update()
    polydata = fill_holes.GetOutput()

    if save_name:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(save_name)
        writer.Write()
    return polydata

def map_scalars_from_source_to_target(mesh_inner, mesh_outer, vectors, source_values, save_name=None, target_array_name="scalars"):
    mesh_vectors = vtk.vtkPolyData()
    end_points = vtk.vtkPoints()
    for i in range(mesh_inner.GetNumberOfPoints()):
        end_points.InsertNextPoint(vectors[i][1] + vectors[i][2])
    mesh_vectors.SetPoints(end_points)
    locator_end_points = vtk.vtkPointLocator()
    locator_end_points.SetDataSet(mesh_vectors)
    locator_end_points.BuildLocator()
    target_values = vtk.vtkFloatArray()
    target_values.SetNumberOfComponents(1)
    target_values.SetName(target_array_name)
    for i in range(mesh_outer.GetNumberOfPoints()):
        point = mesh_outer.GetPoint(i)
        closest_point_id = locator_end_points.FindClosestPoint(point)
        val = source_values[closest_point_id]
        target_values.InsertNextValue(val)
    mesh_outer.GetPointData().SetScalars(target_values)
    if save_name:
        write_vtk_mesh(mesh_outer, save_name)

def bresenham_line(start, end):
    """Bresenham's Line Algorithm in 3D."""
    points = []
    x1, y1, z1 = start
    x2, y2, z2 = end
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    points.append((x1, y1, z1))
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            points.append((x1, y1, z1))
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            points.append((x1, y1, z1))
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            points.append((x1, y1, z1))
    return points

def get_ED_ES(path, split, df_name="LV_blood_stats.csv", printout=False):
    df_path = os.path.join(path, df_name)
    df = pd.read_csv(df_path)
    vols = []
    for ind in split.index:
    
        scan = split["pseudonymized_id"][ind]
        val = df[df.id_seg==scan]["tot_vol_mm3"]
        vols.append(val.item())

    all_vols = np.array(vols)
    ED = all_vols.argmax()
    ES = all_vols.argmin()

    if printout:
        print(f"ED={ED:02d} - {split.percentage.iloc[ED]:02d}%: {split.pseudonymized_id.iloc[ED]}")
        print(f"ES={ES:02d} - {split.percentage.iloc[ES]:02d}%: {split.pseudonymized_id.iloc[ES]}")

    return ED, ES

def get_EF(path, split, df_name="LV_blood_stats.csv", printout=False):
    df_path = os.path.join(path, df_name)
    df = pd.read_csv(df_path)
    vols = []
    for ind in split.index:
    
        scan = split["pseudonymized_id"][ind]
        val = df[df.id_seg==scan]["tot_vol_mm3"]
        vols.append(val.item())

    all_vols = np.array(vols)
    EF = (max(vols) - min(vols))/max(vols)

    if printout:
        print(f"EF: {EF:.2f}")

    return EF

def sitk2vtk(img, flip_for_volume_rendering=False, debugOn=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # Volume rendering does not support direction matrices (27/5-2023)
    # so sometimes the volume rendering is mirrored
    # this a brutal hack to avoid that
    if flip_for_volume_rendering:
        if direction[4] < 0:
            i2 = np.fliplr(i2)

    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()

    return vtk_image

def resample_to_smallest_spacing(image, smallest_spacing=None):
    # Get the smallest voxel spacing in the image
    smallest_spacing = min(image.GetSpacing()) if smallest_spacing is None else smallest_spacing

    # Define the isotropic spacing
    isotropic_spacing = [smallest_spacing] * image.GetDimension()

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, isotropic_spacing)]

    # Create a reference image with isotropic voxel spacing
    reference_image = sitk.Image(new_size, image.GetPixelIDValue())
    reference_image.SetSpacing(isotropic_spacing)
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    # print(f"Reference image size: {reference_image.GetSize()}")
 
    # Resample the original image to match the spacing of the reference image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_image = resampler.Execute(image)
    # print(f"Resampled image size: {resampled_image.GetSize()}")

    return resampled_image

def convert_label_map_to_surface(label_name, output_file, 
                                 reset_direction_matrix=False, 
                                 segment_id=1,
                                 only_largest_component=True,
                                 resample_to_isotropic=False,
                                 exist_ok=False):
    
    if exist_ok and os.path.isfile(output_file):
        return True

    try:
        img = sitk.ReadImage(label_name)
        # Add padding to avoid edge effects
        padding = [1,1,1]
        img = sitk.ConstantPad(img, padding, padding, constant=0)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {label_name}")
        return None
    if resample_to_isotropic:
        img = resample_to_smallest_spacing(img)

    vtk_img = sitk2vtk(img, flip_for_volume_rendering=False)
    if vtk_img is None:
        return False

    # Check if there is any data
    vol_np = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    if np.sum(vol_np) < 1:
        print(f"Only zeros in {label_name}")
        return False

    if reset_direction_matrix:
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        vtk_img.SetDirectionMatrix(direction)

    # print(f"Generating: {output_file}")

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found in {label_name}")
        return False
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(200)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(30.0)
    smoother.SetPassBand(0.01)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update() 

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(15.0)
    normals.Update()
    # Save in VTK version 4.2 and ASCII format so Elastix can read it
    if only_largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(normals.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(conn.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(normals.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()

    return True

def decimate_vtk_mesh(mesh, n_points=None, reduction_factor=None, smooth_fill=True):

    if n_points is not None:
        reduction = 1.0 - n_points / mesh.GetNumberOfPoints()
    elif reduction_factor is not None:
        reduction = reduction_factor
    else:
        raise ValueError("Either n_points or reduction_factor must be provided")

    reduction = 1 - n_points/mesh.GetNumberOfPoints()
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(reduction)
    decimate.Update()

    mesh_reduce = decimate.GetOutput()

    if not smooth_fill:
        return mesh_reduce
    
    # fill holes
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(mesh_reduce)
    fill.SetHoleSize(10000)
    fill.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(fill.GetOutputPort())
    smoother.SetNumberOfIterations(100)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(0.0)
    smoother.SetPassBand(0.01)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update() 

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smoother.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    # normals.SetFeatureAngle(15.0)
    normals.Update()

    return normals.GetOutput()

def decimate_vtk_mesh_paths(mesh_path, save_name, n_points=None, reduction_factor=None, smooth_fill=True):

    mesh = read_vtk_mesh(mesh_path)
    mesh_reduce = decimate_vtk_mesh(mesh, n_points, reduction_factor, smooth_fill)
    if save_name:
        write_vtk_mesh(mesh_reduce, save_name)
    return mesh_reduce

def set_scalars_on_vtk_mesh(mesh, scalars, array_name="scalars"):
    if isinstance(scalars, list):
        scalars = np.array(scalars)
    if isinstance(scalars, vtk.vtkDataArray):
        scalars_vtk = scalars
    else:
        scalars_vtk = numpy_to_vtk(scalars, deep=True)
    scalars_vtk.SetName(array_name)
    
    if len(scalars) == mesh.GetNumberOfPoints():
        mesh.GetPointData().SetScalars(scalars_vtk)
    elif len(scalars) == mesh.GetNumberOfCells():
        mesh.GetCellData().SetScalars(scalars_vtk)
        cell_to_point_filter = vtk.vtkCellDataToPointData()
        cell_to_point_filter.SetInputData(mesh)
        cell_to_point_filter.Update()
        mesh = cell_to_point_filter.GetOutput()
    else:
        raise ValueError("Number of scalars must match number of points or cells")
    return mesh

def set_scalars_on_vtk_mesh_paths(mesh_path, save_name, scalars, array_name="scalars"):
    mesh = read_vtk_mesh(mesh_path)
    mesh = set_scalars_on_vtk_mesh(mesh, scalars, array_name)
    write_vtk_mesh(mesh, save_name)
    return mesh

def print_sitk_info(image):
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    direction = np.array(image.GetDirection()).reshape(3,3)
    print(f"Direction: {direction}")
    print(f"Number of components: {image.GetNumberOfComponentsPerPixel()}")
    print(f"Pixel type: {image.GetPixelIDTypeAsString()}")



def create_gif(filenames, save_folder, save_name, delete_files=False):
    save_name = save_name if save_name.endswith(".gif") else save_name + ".gif"
    with imageio.get_writer(os.path.join(save_folder, save_name), fps=8, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if delete_files:
                os.remove(filename)
    print(f"GIF saved as {save_name}")
    
    
def darken_cmap(cmap, factor=0.85):
    """Darken a colormap by a given factor.
    factor < 1 makes the colormap darker, factor > 1 makes it lighter."""
    cmap = plt.get_cmap(cmap)
    new_cmap = cmap(np.linspace(0, 1, cmap.N))
    # Darken each RGB color
    new_cmap[:, :3] = new_cmap[:, :3] * factor
    return mcolors.ListedColormap(new_cmap)

def plot_values_over_cardiac_cycle(values, save_folder, name, y_label, interp=False):
    fig, ax = plt.subplots( figsize=(10, 6))
    if interp:
        x, y = spline_interp(values.mean(1))
        ax.plot(x, y)
    else:
        ax.plot(values.mean(1))
    ax.fill_between(
        np.arange(values.shape[0]),
        values.mean(1) - values.std(1),
        values.mean(1) + values.std(1),
        alpha=0.1,
        color="blue")
    ax.set_xticks(np.arange(0,20,2))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=20.0))
    ax.set_ylabel(y_label)
    ax.set_title(name)
    ax.set_xlabel("Scan percentage")
    plt.savefig(os.path.join(save_folder, f"{name.replace(' ','_')}.svg"))
    
    
def save_vtk_lines(data, save_name, ref=None):
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    
    t, n, _ = data.shape
    
    colormap = cm.get_cmap('viridis', t) if save_name.find('GT') != -1 else cm.get_cmap('twilight', t) 
    norm = Normalize(vmin=0, vmax=t-1)
    colors = [colormap(norm(i)) for i in range(t)] 
    line_colors = vtk.vtkUnsignedCharArray()
    line_colors.SetNumberOfComponents(3)  # RGB only (no alpha)
    line_colors.SetName("Colors")
    
    if ref is not None:
        data = np.array([ref.TransformContinuousIndexToPhysicalPoint(data[i,j]) for i in range(t) for j in range(n)]).reshape(t,n,3)

    # Add points and create lines for each trajectory
    for i in range(n):  # Loop over n points
        point_ids = []
        for j in range(t):  # Loop over timesteps
            # Insert each point over time and store its ID
            point_id = points.InsertNextPoint(data[j, i])
            point_ids.append(point_id)

        # Connect consecutive points with vtkLine
        for k in range(len(point_ids) - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, point_ids[k])       # Start of the line
            line.GetPointIds().SetId(1, point_ids[k + 1])   # End of the line
            lines.InsertNextCell(line)
            
            rgba = colors[k]  # RGBA tuple from colormap
            rgb = [int(255 * c) for c in rgba[:3]]  # Convert to 0-255 range
            line_colors.InsertNextTuple(rgb)

    # Create vtkPolyData to store points and lines
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    
    # Add the colors to the polydata
    poly_data.GetCellData().SetScalars(line_colors)

    # Write the polydata to a file compatible with Slicer
    writer = vtk.vtkXMLPolyDataWriter()  # For .vtp files
    writer.SetFileName(save_name)   
    writer.SetInputData(poly_data)
    writer.SetDataModeToAscii()  # Ensure compatibility with Slicer
    writer.Write()

