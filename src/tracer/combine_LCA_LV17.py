import os
import re
import numpy as np
import SimpleITK as sitk
import json
import vtk
from scipy.spatial import KDTree
from skimage.morphology import binary_dilation
from scipy.ndimage import center_of_mass
from platipy.imaging.label.utils import get_com
from platipy.imaging.utils.crop import crop_to_roi, label_to_roi
from platipy.imaging.utils.geometry import vector_angle
from platipy.imaging.utils.valve import generate_valve_using_cylinder
from pathlib import Path

from bullseye import create_single_bs_from_mesh
import utils

def generate_lvm_17_mesh(path,segmentation_path):
    #label_myo_path = os.path.join(path, "segmentations", "total_seg", "total_seg.nii.gz")
    os.makedirs(os.path.join(path, "surfaces"), exist_ok=True)
    mesh_path = os.path.join(path, "surfaces", "myocardium.vtk")
    utils.convert_label_map_to_surface(segmentation_path, mesh_path, segment_id=1)

    mesh = utils.read_vtk_mesh(mesh_path)
    mesh = utils.decimate_vtk_mesh(mesh, 20000)
    lv17_path = os.path.join(path, "segmentations","lv17","lv17.nii.gz")
    lv17 = sitk.ReadImage(lv17_path)
    lv_np = sitk.GetArrayFromImage(lv17).transpose(2,1,0)
    labeled_pixels = np.argwhere(lv_np)
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(labeled_pixels)
    # loop though points in mesh
    scalars = np.zeros(mesh.GetNumberOfPoints())
    for i in range(mesh.GetNumberOfPoints()):
        point = mesh.GetPoint(i)
        index = lv17.TransformPhysicalPointToContinuousIndex(point)
        dist, idx = tree.query(index)
        scalars[i] = lv_np[tuple(labeled_pixels[idx])]
        # scalars[i] = lv17[[idx[0], idx[1], idx[2]]]
    # sampler = utils.read_vtk_and_get_sampler(os.path.join(path,r"segmentations\lv17\lv17.nii.gz"), use_nn=True)

    mesh = utils.set_scalars_on_vtk_mesh(mesh, scalars)
    return mesh, scalars
    #utils.write_vtk_mesh(mesh, os.path.join(path, "surfaces", "myocardium_17.vtk"))


def show_lv17_segment_on_LCA(lv17_mesh, lv17_scalars, ca_traced_path):
    ca_traced = utils.read_vtk_mesh(ca_traced_path)
    ca_scalars = ca_traced.GetPointData().GetScalars()

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(lv17_mesh)
    locator.BuildLocator()
    
    closest_idx_lv17 = np.zeros(ca_traced.GetNumberOfPoints(), dtype=int)
    
    for i in range(ca_traced.GetNumberOfPoints()):
        point = ca_traced.GetPoint(i)
        idx = locator.FindClosestPoint(point)
        ca_scalars.SetValue(i, lv17_scalars[idx])
        closest_idx_lv17[i] = idx
    ca_traced.GetPointData().SetScalars(ca_scalars) 
    return ca_traced, closest_idx_lv17
        
def add_LCA_onto_lv17_mesh(lv17_mesh, closest_idx_lv17):
    lv17_scalars = lv17_mesh.GetPointData().GetScalars()
    for i in range(len(closest_idx_lv17)):
        idx = closest_idx_lv17[i]
        lv17_scalars.SetValue(idx,18)
    lv17_mesh.GetPointData().SetScalars(lv17_scalars) 
    return lv17_mesh
    
   
   

if __name__ == "__main__":

    id = "0010"
    series_id = "0036"
    working_dir = Path.cwd()

    # Construct the paths dynamically using pathlib
    lv17_surface_path = working_dir / f'assets/data/{id}/processed/surfaces/myocardium_17.vtk'
    segmentation_path = working_dir / f'assets/data/{id}/raw/CFA-PILOT_{id}_SERIES{series_id}_labels.nii.gz'
    ca_traced_path = working_dir / f"assets\data\CoronaryTracing\CFA-PILOT_{id}_SERIES{series_id}\path_tracing\combined_paths\CFA-PILOT_{id}_SERIES{series_id}_combined_tree_spline.vtk"
    #image_path = working_dir / f'assets/data/{id}/raw/CFA-PILOT_{id}_SERIES{series_id}.nii.gz'
    
    # # Project the LV17 segmentation onto the LCA traced segments
    # lv_mesh, s = generate_lvm_17_mesh(path=working_dir / f'assets/data/{id}/processed', segmentation_path=segmentation_path)
    # ca_traced, idx_lv17 = show_lv17_segment_on_LCA(lv_mesh, s, ca_traced_path)
    # utils.write_vtk_mesh(ca_traced, working_dir / f"assets/data/{id}/processed/surfaces/lv17_on_LCA_{id}.vtk")

    # # Projects the LCA traced segments onto the LV17 mesh
    # lv17_with_LCA = add_LCA_onto_lv17_mesh(lv_mesh, idx_lv17)
    # utils.write_vtk_mesh(lv17_with_LCA, working_dir / f"assets/data/{id}/processed/surfaces/LV17_CA_combined_{id}.vtk")
    
    # Create bullseye plot of the combined mesh
    folder = working_dir / f'assets/data/{id}'
    plot_folder = working_dir / f'assets/data/{id}/bullseye'
    create_single_bs_from_mesh(split=0, folder=str(folder), mesh_path=str(working_dir / f'assets/data/{id}/processed/surfaces/LV17_CA_combined_{id}.vtk'), plot_folder=str(plot_folder), idx=None, id=id,series_id=series_id)