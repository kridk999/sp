import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import vtk
#from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import tqdm
import SimpleITK as sitk
import json
from skimage import morphology, measure
from scipy.spatial import cKDTree
import shutil
from pathlib import Path

import utils


def get_scalar_ring_mm_coordinates(path, mesh_name, exists_ok=True, lvm=True, id='0010', series_id='0020'):
    total_path = os.path.join(path, f"raw/CFA-PILOT_{id}_SERIES{series_id}_labels.nii.gz")
    label_total = sitk.ReadImage(total_path)
    label_lv = sitk.Or(label_total == 1, label_total == 5) if lvm else label_total == 1

    im_bin = sitk.GetArrayFromImage(label_lv).transpose(2, 1, 0)
    im_ring = im_bin & ~morphology.binary_erosion(im_bin)
    im_ring = im_ring.astype(np.uint8)

    mesh = utils.read_vtk_mesh(mesh_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    radius = 0.5 # mm

    ring_nnz = im_ring.nonzero()
    scalar_ring = np.zeros_like(im_ring, dtype=np.float64)
    for x, y, z in zip(*ring_nnz):
        indices = np.array([x, y, z], dtype=np.float64)
        point = label_total.TransformContinuousIndexToPhysicalPoint(indices)
        ids = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, point, ids)
        if ids.GetNumberOfIds() == 0:
            idx = locator.FindClosestPoint(point)
            scalar_ring[x, y, z] = mesh.GetPointData().GetScalars().GetTuple1(idx)
        else:
            scalar_ring[x, y, z] = np.mean([mesh.GetPointData().GetScalars().GetTuple1(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
            # print(ids.GetNumberOfIds())
        scalar_ring[x, y, z] += 1e-10 # to avoid 0 values
        
    scalar_image = sitk.GetImageFromArray(scalar_ring.transpose(2, 1, 0))
    scalar_image.CopyInformation(label_total)
    sitk.WriteImage(scalar_image, os.path.join(path, "processed/segmentations", f"mm_{os.path.basename(mesh_name).split('.')[0]}_ring.nii.gz"))
    return scalar_image

def get_scalar_ring_points_ids(path, mesh_name, exists_ok=True, id='0010', series_id='0020'):
    total_path = os.path.join(path, f"raw/CFA-PILOT_{id}_SERIES{series_id}_labels.nii.gz")
    label_total = sitk.ReadImage(total_path)
    label_lv = sitk.Or(label_total == 1, label_total == 3)

    im_bin = sitk.GetArrayFromImage(label_lv).transpose(2, 1, 0)
    im_ring = im_bin & ~morphology.binary_erosion(im_bin)
    im_ring = im_ring.astype(np.uint8)

    mesh = utils.read_vtk_mesh(mesh_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    radius = 0.25 # mm

    ring_nnz = im_ring.nonzero()
    scalar_ring_ids = {} #np.zeros_like(im_ring, dtype=np.float64)
    for x, y, z in zip(*ring_nnz):
        indices = np.array([x, y, z], dtype=np.float64)
        point = label_total.TransformContinuousIndexToPhysicalPoint(indices)
        ids = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, point, ids)
        if ids.GetNumberOfIds() == 0:
            idx = locator.FindClosestPoint(point)
            scalar_ring_ids[(x, y, z)] = [idx]
        else:
            scalar_ring_ids[(x, y, z)] = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]

    return scalar_ring_ids

def get_scalar_values_from_ring_points(path, mesh_name, scalar_ring_ids, reference):
    
    mesh = utils.read_vtk_mesh(mesh_name)
    scalar_ring = np.zeros(len(scalar_ring_ids), dtype=np.float64)
    ref = sitk.GetArrayFromImage(reference).transpose(2, 1, 0)
    scalar_ring = np.zeros_like(ref, dtype=np.float64)
    
    for i, (x, y, z) in enumerate(scalar_ring_ids.keys()):
        ids = scalar_ring_ids[(x, y, z)]
        scalar_ring[x, y, z] = np.mean([mesh.GetPointData().GetScalars().GetTuple1(i) for i in ids])
        scalar_ring[x, y, z] += 1e-10
        
    scalar_image = sitk.GetImageFromArray(scalar_ring.transpose(2, 1, 0))
    scalar_image.CopyInformation(reference)
    return scalar_image
    
            
def generate_polar_values(path,
                          mesh_name,
                          scalar_ring_ids=None,
                          return_xy=True,
                          exists_ok=True,
                          lvm=True,
                          id='0010',
                          series_id='0020'):
    """
      1. read dist_mesh with vtk
      2. convert mesh to segmentation with one ring of values
      3. read transform
      4. resample segmentation with transform
      5. generate values from polar coordinates
    """    

    input_path = os.path.join(path, f"raw/CFA-PILOT_{id}_SERIES{series_id}_labels.nii.gz")
    label_total = sitk.ReadImage(input_path)
    lab_total = sitk.GetArrayFromImage(label_total).transpose(2, 1, 0)
    ignore_ring = morphology.binary_dilation(lab_total == (3 if lvm else 1)) #outer_ring & inner_ring
    
    if scalar_ring_ids is not None:
        scalar_image = get_scalar_values_from_ring_points(path, mesh_name, scalar_ring_ids, label_total)
    else:
        scalar_image = get_scalar_ring_mm_coordinates(path, mesh_name, exists_ok=exists_ok, lvm=lvm, id=id, series_id=series_id)

    sitk.WriteImage(scalar_image, os.path.join(path, "processed/segmentations", f"mm_{os.path.basename(mesh_name).split('.')[0]}_ring.nii.gz"))
    # read transform
    transform_path = os.path.join(path, "processed/lv17_transform.txt")
    transform = sitk.ReadTransform(transform_path)

     # read json
    json_path = os.path.join(path, "processed/segmentations", "lv17", "lv17.json")
    with open(json_path, "r") as f:
        output_constants = json.load(f)
    ranges = output_constants["ranges"]
    angle_offset = output_constants["theta0"] -2/3*np.pi #- np.pi/6 -np.pi/24 #- 2/3 * np.pi
    y_0, x_0 = output_constants["com_yx_short_long_axis"]


    label_lv = sitk.Or(label_total == 1, label_total == 3) if lvm else label_total == 1
    
    lv_transformed = sitk.Resample(label_lv,
                             transform,
                             sitk.sitkNearestNeighbor,
                             0,
                             scalar_image.GetPixelID())
    com = measure.centroid(sitk.GetArrayFromImage(lv_transformed).transpose(2, 1, 0))
    x_0, y_0 = com[0], com[1]

    # resample segmentation with transform
    scalar_im = sitk.GetArrayFromImage(scalar_image).transpose(2, 1, 0)
    #scalar_im[~ignore_ring] = 0 
    scalar_image = sitk.GetImageFromArray(scalar_im.transpose(2, 1, 0))
    scalar_image.CopyInformation(label_total)
    scalar_seg = sitk.Resample(scalar_image,
                             transform,
                             sitk.sitkNearestNeighbor,
                             0,
                             scalar_image.GetPixelID())

    # generate values from polar coordinates
    scalar_np = sitk.GetArrayFromImage(scalar_seg).transpose(2, 1, 0)
    slices_nnz = scalar_np.nonzero()[2]
    slices_ = np.unique(slices_nnz)
    slices_ = slices_[slices_ < ranges[-1]]

    angles, radii, polar_vals = [], [], []

    for n in slices_: 
        
        label_slice = scalar_np[:,:,n]
        loc_x, loc_y = np.where(label_slice)

        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - angle_offset


        vals = label_slice[loc_x, loc_y]
        angles.extend(theta)
        radii.extend([n]*len(theta))
        polar_vals.extend(vals)
        
        # plt.figure()
        # plt.scatter(loc_x, loc_y, c=vals, cmap="plasma")
        # plt.plot(x_0, y_0, 'ro')
        # plt.savefig(f"slice_{n}.png")
        
    if return_xy:
        radii = np.array(radii) - min(radii)
        x, y = utils.polar2cartesian(radii, np.array(angles))
        sort_radii = np.argsort(radii)[::-1]
        x, y, polar_vals = x[sort_radii], y[sort_radii], np.array(polar_vals)[sort_radii]
        return x, y, polar_vals
    else:
        return angles, radii, polar_vals
    
    
def fill_holes_in_bullseye(x, y, polar_vals, n_points_per_ring=500, n_neighbors=5):
    # make default grid
    radius = np.max(np.sqrt(x**2 + y**2))
    theta = np.linspace(0, 2*np.pi, n_points_per_ring)
    # make meshgrid
    Th, Rad = np.meshgrid(theta, np.linspace(0, radius, n_points_per_ring))
    # loop through grid nad find n closest points
    # using kd tree

    # r, t = utils.cartesian2polar(x, y)
    # points = np.array([r, t]).T
    points = np.array([x, y]).T
    tree = cKDTree(points)
    polar_output = np.zeros_like(Th)
    X, Y = utils.polar2cartesian(Rad.ravel(), Th.ravel())
    points_regular = np.array([X, Y]).T
    dist, ind = tree.query(points_regular, k=n_neighbors)
    polar_output = np.median(polar_vals[ind], axis=1)

    #  = utils.polar2cartesian(Rad.flatten(), Th.flatten())
    mask = ~ (dist.mean(axis=1) > 2)
    X, Y, polar_output = X[mask], Y[mask], polar_output[mask]
    return X, Y, polar_output

def create_single_bs_from_mesh(split, folder, mesh_path, plot_folder, idx, scalar_ring_ids=None, global_min=None, global_max=None, id='0010', series_id='0020'):
    x, y, polar_vals = generate_polar_values(Path(folder), mesh_path, scalar_ring_ids=scalar_ring_ids, id=id, series_id=series_id)
    x, y, polar_vals = fill_holes_in_bullseye(x, y, polar_vals, 500, 5)
    # x = -x
    # rad, theta = utils.cartesian2polar(x, y)
    # theta -= np.pi/2-np.pi/10
    # x, y = utils.polar2cartesian(rad, theta)
    
    cmap = "plasma" if mesh_path.find("squeez")!=-1 else plt.cm.tab20 if mesh_path.find("17")!=-1 else "seismic"
    # original cmap virker ikke for windows 
    #cmap = "plasma" if "squeez" in str(mesh_path) else plt.cm.tab20 if "17" in str(mesh_path) else "seismic"
    darkened_cmap = utils.darken_cmap(cmap, factor=0.85)
    global_min = np.quantile(polar_vals, 0.001) if global_min is None else global_min
    global_max = np.quantile(polar_vals, 0.999) if global_max is None else global_max
    norm = Normalize(vmin=global_min, vmax=global_max, clip=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(x, y, c=polar_vals, marker='.', linewidths=1, norm=norm)
    # sc = ax.scatter(rad, theta, c=polar_vals, marker='.', linewidths=1, norm=norm)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    sc.set_cmap(darkened_cmap)
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title("ES - Squeez")
    fig.tight_layout()
    os.makedirs(plot_folder, exist_ok=True)
    filename = os.path.join(plot_folder, f"{os.path.basename(mesh_path.split('.')[0])}.png")
    
    plt.savefig(filename)
    plt.close()

def generate_bs_gif(split, folder, save_folder, mesh_name, gif_name=None, global_min=None, global_max=None):
    plot_folder = os.path.join(save_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    scalar_ring_ids = get_scalar_ring_points_ids(os.path.join(folder, split.pseudonymized_id.iloc[0]), os.path.join(save_folder, "meshes", f"{mesh_name}_00.vtk"))
    for i in range(len(split)):
        mesh_name_i = os.path.join(save_folder, "meshes", f"{mesh_name}_{i*5:02d}.vtk")
        create_single_bs_from_mesh(split, folder, mesh_name_i, plot_folder, i, scalar_ring_ids=scalar_ring_ids, global_min=global_min, global_max=global_max)
    
    _, ES = utils.get_ED_ES(folder, split)
    shutil.copy2(os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_{ES*5:02d}.png"),
                 os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_ES.png"))
    
    filenames = [os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_{i*5:02d}.png") for i in range(len(split))]
    gif_name = gif_name if gif_name is not None else f"{mesh_name.split('.')[0]}_bs.gif"
    utils.create_gif(filenames, os.path.join(save_folder, "gifs"), gif_name, delete_files=True)
    



if __name__ == "__main__":

    id = '0010'
    series_id = '0020'
    
    working_dir = Path.cwd()

    folder = working_dir / f'assets/data/{id}'
    plot_folder = working_dir / f'assets/data/{id}/bullseye'

    create_single_bs_from_mesh(split=0, folder=str(folder), mesh_path=str(working_dir / f'assets/data/{id}/processed/surfaces/myocardium_17.vtk'), plot_folder=str(plot_folder), idx=None)