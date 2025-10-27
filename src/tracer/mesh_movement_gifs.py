import os
import numpy as np
import pyvista as pv
import imageio
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
import seaborn as sns

#from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from utils import general, utils



def bbox(img):
    img = (img != 255)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
    return rmin, rmax, cmin, cmax

def crop_frames_from_biggest_bbox(frames, buffer=10):
    bbs = np.array([bbox(f.mean(-1)) for f in frames])
    bb = (bbs[:,0].min() - buffer, 
          bbs[:,1].max() + buffer, 
          bbs[:,2].min() - buffer, 
          bbs[:,3].max() + buffer)
    
    return [f[bb[0]:bb[1], bb[2]:bb[3]] for f in frames]

def render_and_save_frames(save_folder, mesh_name, reference, gif_filename=None, fps=8, view_front=True, gif_type="strain", cmap="seismic", rv_folder=None, split=None): #TODO: add color map as argument
    """
    Render the sequence of meshes without rotation and save them as a GIF.
    """
    
    mesh_names = glob.glob(os.path.join(save_folder, "meshes", f"{mesh_name}_[0-9][0-9].vtk"))
    mesh_names.sort()
    
    if rv_folder is not None:
        use_rv = True
        rv_names = []
        for i in range(len(mesh_names)):
            total_seg_path = os.path.join(rv_folder, split.pseudonymized_id.iloc[i], "segmentations/total_seg/total_seg.nii.gz")
            rv_path = os.path.join(rv_folder, split.pseudonymized_id.iloc[i], "surfaces/rv.vtk")
            rv_names.append(rv_path)
            utils.convert_label_map_to_surface(total_seg_path, rv_path, segment_id=2)
    else:
        use_rv = False
        
    if gif_filename is None:
        gif_filename = f"{mesh_name}_{'front' if view_front else 'back'}.gif"
    gif_filename = gif_filename.replace(".gif", "_rv.gif") if use_rv else gif_filename
    
    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=True, window_size=(1600,1600))
    
    plotter.background_color = 'white'
    # if view_front:
    plotter.camera_position =[(0, 0.4, 2.5), (0, 0, 0), (-0.4, -0.3, 0.5)] if view_front else [(-0.9,-0.7, -1.6), (0, 0, 0), (-0.4, -0.3, 0)]
    # plotter.camera_position =[(0.2, 1, 2), (0,0,0), (-0.2, -0.2, 0.7)] if view_front else [(-0.2, -1, -2), (0,0,0), (-0.2, -0.2, 0.7)]
    # else:
    #     plotter.camera_position =[(-0.2, -1, -2), (0,0,0), (-0.2, -0.2, 0.7)]

    
    sargs = dict(height=0.5, vertical=True, position_x=0.2, position_y=0.2, n_colors=20)

    # Animation parameters
    frames = []  # To store GIF frames
    mesh = pv.read(mesh_names[0])
    clim = [0.75,1] if gif_type=="squeez" else [-0.3,0.3]
    cmap = "plasma" if gif_type=="squeez" else "seismic"
    
    # Render each mesh
    for i, name in enumerate(mesh_names):
        # Load the mesh
        mesh_vtk = utils.read_vtk_mesh(name)
        if gif_type == "GT":
            mesh = pv.read(name)
            points = mesh.points
        else:
            points = vtk_to_numpy(mesh_vtk.GetPoints().GetData())

        #TODO: centralize mesh in 0,0,0
        
        mesh.points = general.scale_points_from_reference_to_1_1(points, reference)
        mesh_mean = np.mean(mesh.points, axis=0)
        mesh.points = mesh.points - mesh_mean
        mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
            
        if gif_type in ["squeez", "strain"]:
            if mesh_vtk.GetCellData().GetScalars():
                mesh.cell_data.set_scalars(mesh_vtk.GetCellData().GetScalars())
            elif mesh_vtk.GetPointData().GetScalars():
                mesh.point_data.set_scalars(mesh_vtk.GetPointData().GetScalars())
            plotter.add_mesh(mesh, style='surface', show_edges=True, edge_color='black', lighting=True, scalar_bar_args=sargs, clim=clim, cmap=cmap) 
            plotter.remove_scalar_bar()
        elif gif_type == "17seg":    
            mesh.point_data.set_scalars(mesh_vtk.GetPointData().GetScalars())
            tab20_cmap = plt.cm.get_cmap("tab20")
            # Create a custom colormap that includes white for scalar -1
            custom_colors = np.zeros((18, 4))  # RGBA (18 rows: -1, 1-17)
            custom_colors[1:] = tab20_cmap(np.arange(17))  # Add Tab20 colors
            custom_colors[:1] = [1, 1, 1, 1]  # White for scalar -1
            custom_cmap = ListedColormap(custom_colors)
            plotter.add_mesh(mesh, interpolate_before_map=False, style='surface', show_edges=True, edge_color='black', lighting=True, scalar_bar_args=sargs, cmap=custom_cmap, clim=[0, 17])
            plotter.remove_scalar_bar()
        elif gif_type == "stripes":
            if i==0:
                stripes = np.sin(0.2 * points[:, 1])
            mesh.point_data.set_scalars(stripes)
            plotter.add_mesh(mesh, style='surface', show_edges=True, edge_color='black', lighting=True)
            plotter.remove_scalar_bar()
        elif gif_type == "GT":
            stripes = np.sin(0.2 * points[:, 1])
            mesh.point_data.set_scalars(stripes)
            actor = plotter.add_mesh(mesh, style='surface', lighting=True)
            plotter.remove_scalar_bar()
       
        if use_rv:
            rv_mesh = pv.read(rv_names[i])
            rv_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
            rv_mesh.points = general.scale_points_from_reference_to_1_1(rv_mesh.points, reference)*0.5 - mesh_mean
            rv_actor = plotter.add_mesh(rv_mesh, style='surface',color='purple', opacity=0.4, lighting=True)


        # Render and capture the frame
        img = plotter.screenshot(return_img=True)
        frames.append(np.fliplr(img))
        
        # remove actors
        if gif_type == "GT":
            plotter.remove_actor(actor)
        if use_rv:
            plotter.remove_actor(rv_actor)

    os.makedirs(os.path.join(save_folder, "gifs"), exist_ok=True)
    frames_cropped = crop_frames_from_biggest_bbox(frames, buffer=50)
    
    # save frames as png
    os.makedirs(os.path.join(save_folder, "gifs", "frames"), exist_ok=True)
    for i, frame in enumerate(frames_cropped):
        plt.imsave(os.path.join(save_folder, "gifs", "frames", f"{gif_filename[:-4]}_{i*5:02d}.png"), frame)
    
    
    imageio.mimsave(os.path.join(save_folder, "gifs", gif_filename), frames_cropped, fps=8, loop=0)
    
    print(f"GIF saved as {gif_filename}")