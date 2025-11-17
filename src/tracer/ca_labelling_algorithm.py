import numpy as np
import json
import glob
import os
import vtk
from utils import vtk_to_numpy
import utils


#load point file from json as a dictionary
def load_points_from_json(file_path):
    with open(file_path, 'r') as f:
        points_dict = json.load(f)
    for key in points_dict:
        points_dict[key] = np.array(points_dict[key])
        if key == 'start_point':
            points_dict[key] = points_dict[key][0]  
    return points_dict

def distance_between_all_points(points, verbose=False):
    '''Calculate distances between start point, breakpoints, and end points.'''
    start_to_bp_distances = np.linalg.norm(points['all_branch_points'] - points['start_point'], axis=1)
    bp_to_end_distances = [np.linalg.norm(bp - points['end_points'], axis=1) for bp in points['all_branch_points']]
    if verbose:
        print("Distances from start point to breakpoints:", start_to_bp_distances)
        print("Distances from breakpoints to end points:", bp_to_end_distances)
    return start_to_bp_distances, bp_to_end_distances

def distance_between_point_and_set(point, point_set):
    '''Calculate distances between a single point and a set of points.'''
    distances = np.linalg.norm(point_set - point, axis=1)
    return distances

def vector_between_startpoint_and_closest_breakpoint(start_point, start_to_bp_distances, breakpoints):
    '''Get vector from start point to closest breakpoint.'''
    closest_bp_index = np.argmin(start_to_bp_distances)
    closest_bp = breakpoints[closest_bp_index]
    vector = closest_bp - start_point
    return vector, closest_bp_index

def vectors_between_breakpoints(breakpoints):
    vectors = []
    num_bps = breakpoints.shape[0]
    for i in range(num_bps):
        for j in range(num_bps):
            if i != j:
                vector = breakpoints[j] - breakpoints[i]
                vectors.append((breakpoints[j],vector))
    return np.array(vectors)

def remove_vectors_with_positive_dot_product(vectors, start_vector):
    possible_vectors = []
    for bp, vector in vectors:
        dot_product = np.dot(vector, start_vector)
        if dot_product <= 0:
            possible_vectors.append((bp, vector))
    return np.array(possible_vectors)

def load_vtk_trees_without_spline(folder_path):
    tree_files = glob.glob(os.path.join(folder_path, "*path.vtk"))
    trees = []
    for file in tree_files:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        mesh = reader.GetOutput()
        points = vtk_to_numpy(mesh.GetPoints().GetData())
        trees.append(points)
    return trees

def load_vtk_combined_tree(file_path):
    mesh = utils.read_vtk_mesh(file_path)
    return mesh

def get_closest_segment_key_to_tree_point(tree_points, atlas):
    closest_segments = set()
    # make step size 5
    for point in tree_points[::1]:
        com_coordinatres = np.array([coor for coor in atlas])
        distances = np.linalg.norm(com_coordinatres - point, axis=1)
        closest_index = np.argmin(distances)
        closest_segments.add(closest_index + 1) 
    return closest_segments

def get_closest_segment_to_end_point(end_point, atlas):
    com_coordinatres = np.array([coor for coor in atlas])
    distances = np.linalg.norm(com_coordinatres - end_point, axis=1)
    closest_index = np.argmin(distances)
    return closest_index + 1

def label_artery_segment_given_main_branch(tree, point_data, main_tree='LAD', all_trees=None):
    """
    Label an artery segment based on anatomical heuristics.
    
    Parameters:
    -----------
    tree : np.ndarray
        Coordinates of the traced artery path (N x 3)
    point_data : dict
        Contains 'start_point', 'all_branch_points', 'end_points', 'atlas_points'
    main_tree : str
        Either 'LAD' or 'LCX' - the main coronary branch this belongs to
        
    Returns:
    --------
    str : Label like 'LAD', 'LCX', 'Diagonal', 'Obtuse Marginal', or 'Outlier'
    """
    
    # =========================================================================
    # STEP 0: IDENTIFY AND SKIP LMCA PORTION
    # =========================================================================

    # The LMCA is typically 5-15mm. Find where trees start to diverge.
    # The first branch point should be the LAD/LCX bifurcation
    _, lmca_bifurcation_idx = vector_between_startpoint_and_closest_breakpoint(
        point_data['start_point'],
        distance_between_point_and_set(point_data['start_point'], point_data['all_branch_points']),
        point_data['all_branch_points'])
    lmca_bifurcation = point_data['all_branch_points'][lmca_bifurcation_idx]
    
    # Find where this tree reaches the bifurcation
    lmca_end_idx = None
    for i, point in enumerate(tree):
        if np.linalg.norm(point - lmca_bifurcation) < 3.0:
            lmca_end_idx = i
            break

    if lmca_end_idx is None:
        # If we can't find bifurcation, assume first 10% is LMCA
        lmca_end_idx = max(1, len(tree) // 10)

    # Use the post-LMCA portion for analysis
    tree_post_lmca = tree[lmca_end_idx:]
    effective_start = tree[lmca_end_idx]

    # =========================================================================
    # STEP 1: EXTRACT KEY ANATOMICAL FEATURES (UPDATED)
    # =========================================================================
    
    # 1.1 Get tree endpoints - NOW starting after LMCA
    tree_start = effective_start  # After LMCA bifurcation
    tree_end = tree[-1]
    
    # 1.2 Calculate tree length (EXCLUDING LMCA)
    segment_lengths = np.linalg.norm(np.diff(tree_post_lmca, axis=0), axis=1)
    total_length = np.sum(segment_lengths)
    
    # 1.3 Find which branch points lie on or near this tree
    branch_points_on_tree = []
    for bp in point_data['all_branch_points']:
        min_dist_to_tree = np.min(np.linalg.norm(tree_post_lmca - bp, axis=1))
        if min_dist_to_tree < 2.0:  # Threshold in mm
            branch_points_on_tree.append(bp)
    
    # 1.4 Identify which AHA segments the tree passes through
    segments_traversed = get_closest_segment_key_to_tree_point(
        tree, point_data['atlas_points'][()]
    )
    
    # 1.5 Identify termination segment
    end_segment = get_closest_segment_to_end_point(
        tree_end, point_data['atlas_points'][()]
    )
    
    # =========================================================================
    # STEP 2: CALCULATE GEOMETRIC FEATURES (UPDATED)
    # =========================================================================

    # 2.1 Main trunk direction - FROM LMCA bifurcation
    # For LAD: should point anteriorly and apically
    # For LCX: should point laterally (left)
    main_trunk_vector = lmca_bifurcation - point_data['start_point']
    main_trunk_vector_norm = main_trunk_vector / np.linalg.norm(main_trunk_vector)

    # 2.2 This tree's direction - FROM bifurcation, not from aorta
    tree_direction_vector = tree_end - effective_start
    tree_direction_vector_norm = tree_direction_vector / np.linalg.norm(tree_direction_vector)

    # 2.3 For main vessels, check if they CONTINUE from the bifurcation
    # rather than branching off at a sharp angle
    alignment_score = np.dot(main_trunk_vector_norm, tree_direction_vector_norm)
    angle_degrees = np.degrees(np.arccos(np.clip(alignment_score, -1.0, 1.0)))

    # 2.4 Calculate lateral and apical displacement FROM bifurcation
    lateral_displacement = np.abs(tree_end[0] - effective_start[0])

    
    # =========================================================================
    # STEP 3: TERRITORY-BASED SCORING
    # =========================================================================
    
    # Define territories based on main_tree
    if main_tree == 'LAD':
        primary_territory = LAD_TERRITORY
        secondary_territory = LCX_TERRITORY
        main_end_segments = LAD_END_SEGMENTS
    else:  # LCX
        primary_territory = LCX_TERRITORY
        secondary_territory = LAD_TERRITORY
        main_end_segments = LCX_END_SEGMENTS
    
    # 3.1 Calculate territory coverage scores
    primary_score = len(segments_traversed.intersection(primary_territory))
    secondary_score = len(segments_traversed.intersection(secondary_territory))
    territory_ratio = primary_score / (primary_score + secondary_score + 1e-6)
    
    # =========================================================================
    # STEP 4: RULE-BASED CLASSIFICATION LOGIC (UPDATED)
    # =========================================================================

    # Find if this tree branches OFF from main trunk (after bifurcation)
    branches_after_lmca = []
    for bp in point_data['all_branch_points'][1:]:  # Skip first (the bifurcation)
        min_dist = np.min(np.linalg.norm(tree_post_lmca - bp, axis=1))
        if min_dist < 3.0:
            branches_after_lmca.append(bp)

    # Main vessels should:
    # 1. Be long
    # 2. Have multiple branches coming off them
    # 3. NOT diverge sharply from the bifurcation direction
    # 4. End in typical territories

    is_main_trunk = (
        total_length > 100 if main_tree=='LAD' else total_length > 60 and  # Long vessel (excluding LMCA)
        len(branches_after_lmca) >= 1 and  # Has multiple side branches
        angle_degrees < 60 if main_tree=='LAD' else angle_degrees > 60 and  # Continues relatively straight from bifurcation
        end_segment in main_end_segments and
        territory_ratio > 0.65 )
    
    if is_main_trunk:
        return main_tree  # Return 'LAD' or 'LCX'
    
    # 4.2 CHECK IF THIS IS A DIAGONAL (LAD branch)
    if main_tree == 'LAD':
        # Find WHERE along the path this tree diverges (if at all)
        divergence_point = None
        divergence_idx = None

        for i, point in enumerate(tree_post_lmca):
            for bp in point_data['all_branch_points'][1:]:  # Skip LMCA bifurcation
                if np.linalg.norm(point - bp) < 3.0:
                    divergence_point = point
                    divergence_idx = i
                    break
            if divergence_point is not None:
                break

        # Calculate path length from bifurcation to divergence
        if divergence_idx is not None:
            path_to_divergence = np.sum(
                np.linalg.norm(np.diff(tree_post_lmca[:divergence_idx], axis=0), axis=1)
            )
        else:
            path_to_divergence = total_length  # No divergence = probably main trunk

        # Branches should diverge EARLY after bifurcation
        is_diagonal = (
            20 < total_length < 100 and
            divergence_point is not None and
            path_to_divergence < 40 and  # Branches within first 40mm after bifurcation
            angle_degrees > 20 and  # Branches at moderate angle
            lateral_displacement > 10 and
            primary_score > 0 and
            end_segment in {10, 11, 13, 14, 15, 16, 17})
        
        if is_diagonal:
            diagonal_number = determine_diagonal_number(
                tree, 
                point_data['all_branch_points'], 
                lmca_bifurcation,
                all_trees  # Pass all trees
            )
            return f'D{diagonal_number}'
    
    # 4.3 CHECK IF THIS IS AN OBTUSE MARGINAL (LCX branch)
    if main_tree == 'LCX':
        is_marginal = (
            15 < total_length < 110 and  # Medium length
            20 < angle_degrees < 120 and  # Branches at wider angle
            primary_score > 0 and  # Passes through LCX territory
            end_segment in {1, 2, 5, 6, 7, 12} )  # Ends in lateral segments
        
        if is_marginal:
            marginal_number = determine_marginal_number(
                tree,
                point_data['all_branch_points'],
                lmca_bifurcation,
                all_trees  # Pass all trees
            )
            return f'OM{marginal_number}'
    
    # 4.4 CHECK FOR SEPTAL BRANCHES (small, perpendicular, into septum)
    is_septal = (
        total_length < 40 and  # Short
        angle_degrees > 60 and  # Sharp angle
        end_segment in {2, 3, 8, 9, 14}  # Septal segments
    )
    
    if is_septal:
        return 'Septal'
    
    # 4.5 CHECK FOR RAMUS INTERMEDIUS (large branch between LAD/LCX)
    is_ramus = (
        total_length > 30 and
        0.4 < territory_ratio < 0.6 #and  # Mixed territory
        #len(branch_points_on_tree) > 0  # Has its own branches
    )
    
    if is_ramus:
        return 'Ramus Intermedius'
    
    # =========================================================================
    # STEP 5: FALLBACK CLASSIFICATION
    # =========================================================================
    
    # If none of the above, classify as outlier or small branch
    if total_length < 30:
        return 'Small Branch'
    else:
        return 'Outlier'


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def determine_diagonal_number(tree, branch_points, lmca_bifurcation, all_trees):
    global diagonal_count
    dc = diagonal_count + 1
    diagonal_count = dc
    return dc

def determine_marginal_number(tree, branch_points, lmca_bifurcation, all_trees):
    global marginal_count   
    mc = marginal_count + 1
    marginal_count = mc
    return mc

def create_labeled_segments(trees, tree_labels, point_data):
    """
    Create labeled coronary artery segments based on anatomical structure.
    
    Parameters:
    -----------
    trees : list of np.ndarray
        All traced trees (each starting from aorta/LMCA ostium)
    tree_labels : dict
        Labels for each tree (e.g., {'1': 'LAD', '2': 'D1', ...})
    point_data : dict
        Contains 'start_point', 'all_branch_points', 'end_points'
        
    Returns:
    --------
    dict : {label: segment_coordinates}
        e.g., {'LMCA': array(...), 'LAD': array(...), 'D1': array(...)}
    """
    labeled_segments = {}
    
    # =========================================================================
    # 1. CREATE LMCA SEGMENT (from start to first branch point)
    # =========================================================================
    
    # Find the LMCA bifurcation (first/closest branch point to start)
    distances_to_start = np.linalg.norm(
        point_data['all_branch_points'] - point_data['start_point'], 
        axis=1
    )
    lmca_bifurcation_idx = np.argmin(distances_to_start)
    lmca_bifurcation = point_data['all_branch_points'][lmca_bifurcation_idx]
    
    # Use any tree to get the LMCA segment (they're all the same at the start)
    first_tree = trees[0]
    lmca_segment = []
    
    for i, point in enumerate(first_tree):
        lmca_segment.append(point)
        if np.linalg.norm(point - lmca_bifurcation) < 3.0:
            break
    
    labeled_segments['LMCA'] = np.array(lmca_segment)
    
    # =========================================================================
    # 2. CREATE SEGMENTS FOR EACH LABELED TREE
    # =========================================================================
    
    for tree_idx, (tree, label) in enumerate(zip(trees, tree_labels.values())):
        
        # Skip if already processed or unlabeled
        if label in ['Outlier', 'Small Branch']:
            continue
        
        # -----------------------------------------------------------------
        # For LAD and LCX: Use from bifurcation to endpoint
        # -----------------------------------------------------------------
        if label in ['LAD', 'LCX']:
            # Find where this tree reaches the LMCA bifurcation
            bifurcation_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 3.0:
                    bifurcation_idx = i
                    break
            
            if bifurcation_idx is None:
                bifurcation_idx = len(tree) // 10  # Fallback
            
            # Main vessel: from bifurcation to end
            labeled_segments[label] = tree[bifurcation_idx:]
        
        # -----------------------------------------------------------------
        # For branches (Diagonals, Marginals, etc.): Use from last branch point to endpoint
        # -----------------------------------------------------------------
        else:
            # Find where this tree diverges from the main trunk
            # This is the same logic used in label_artery_segment_given_main_branch
            divergence_idx = None
            
            # Skip LMCA portion
            lmca_end_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 3.0:
                    lmca_end_idx = i
                    break
            
            if lmca_end_idx is None:
                lmca_end_idx = len(tree) // 10
            
            tree_post_lmca = tree[lmca_end_idx:]
            
            current_bp_distance = 0.0
            # Find first branch point in post-LMCA portion
            for i, point in enumerate(tree_post_lmca):
                for bp in point_data['all_branch_points'][1:]:  # Skip LMCA bifurcation
                    if np.linalg.norm(point - bp) < 3.0:
                        dist_to_lmca_bifurcation = np.linalg.norm(bp - lmca_bifurcation)
                        if dist_to_lmca_bifurcation > current_bp_distance:
                            divergence_idx = lmca_end_idx + i
                            current_bp_distance = dist_to_lmca_bifurcation
                        #break
                # if divergence_idx is not None:
                #     break
            
            # Use from divergence point to end
            if divergence_idx is not None:
                labeled_segments[label] = tree[divergence_idx:]
            else:
                # Fallback: use from after LMCA
                labeled_segments[label] = tree_post_lmca
            
    return labeled_segments

#utilize this util function: utils.write_vtk_mesh(mesh, path) when saving the combined tree as one vtk file
def save_labeled_segments_to_vtk(labeled_segments, output_folder):
    """
    Save each labeled segment as a separate VTK file.
    
    Parameters:
    -----------
    labeled_segments : dict
        Dictionary from create_labeled_segments()
    output_folder : str
        Folder to save VTK files
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for label, segment in labeled_segments.items():
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        for point in segment:
            vtk_points.InsertNextPoint(point)
        
        # Create polyline
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(segment))
        for i in range(len(segment)):
            polyline.GetPointIds().SetId(i, i)
        
        # Create cells
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyline)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetLines(cells)
        
        # Write to file
        writer = vtk.vtkPolyDataWriter()
        filename = os.path.join(output_folder, f"{label}.vtk")
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        writer.Write()
        
        print(f"Saved {label} segment to {filename}")


if __name__ == "__main__":
    id = '281'
    series_id = '0020'
    Parent_dir = f'assets/data/IMGCAS_tracing/{id}.img'
    point_path = Parent_dir + '/tracer_points.json'
    path_main_tree = Parent_dir + '/tree_angles.json'
    path_trees = Parent_dir + '/path_tracing/combined_paths'
    path_combined_tree = Parent_dir + f'/path_tracing/combined_paths/{id}.img_combined_tree_spline.vtk'
    # path_atlas = f'assets/data/{id}/processed/misc/atlas.json'
    # path_trees = f'assets/data/CoronaryTracing/CFA-PILOT_{id}_SERIES{series_id}/path_tracing/combined_paths'
    points = load_points_from_json(point_path)
    trees = load_vtk_trees_without_spline(path_trees)
    main_trunk = json.load(open(path_main_tree, 'r'))['tree_labels']
    
    # AHA 17-Segment Model Territories
    LAD_TERRITORY = {4,10,11,13,14,15,16,17}    # Anterior, Anteroseptal, Apex
    LCX_TERRITORY = {1,2,5,6,7,12}              # Anterolateral, Inferolateral
 
    # Define key segments for termination
    LAD_END_SEGMENTS = {17, 13, 14,1,7}         # Apex, Apical Anterior/Anteroseptal
    LCX_END_SEGMENTS = {11, 12, 16, 5, 6,1}     # Mid/Apical Lateral
 
    main_art = {k: 'LAD' if v == 'LCA' else 'LCX' for k, v in main_trunk.items()}
    
    diagonal_count = 0
    marginal_count = 0
    tree_data = {}
    end_segments_data = {}
    tree_labels = {}
    for idx, (end_point, tree) in zip(main_art.keys(), zip(points['end_points'], trees)):
        tree_idx_str = idx
        
        tree_label = label_artery_segment_given_main_branch(tree, points, main_art[idx],all_trees=trees)
        tree_labels[tree_idx_str] = tree_label
    print("Classified Tree Labels:", tree_labels)

    labeled_segments = create_labeled_segments(trees, tree_labels, points)
    output_folder = Parent_dir + '/labeled_segments_vtk'
    save_labeled_segments_to_vtk(labeled_segments, output_folder)