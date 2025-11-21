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

def analyze_vessel_direction(main_trunk_vector, tree_direction_vector, tree_end, effective_start):
    """
    Analyze vessel direction using cross product and spatial components.
    
    Returns:
    --------
    dict : Contains directional features for classification
    """
    # Normalize vectors
    main_norm = main_trunk_vector / np.linalg.norm(main_trunk_vector)
    tree_norm = tree_direction_vector / np.linalg.norm(tree_direction_vector)
    
    # Cross product analysis
    cross = np.cross(main_norm, tree_norm)
    
    # Spatial displacement analysis
    displacement = tree_end - effective_start
    
    return {
        # Cross product components
        'cross_lateral': cross[0],      # X: Left(+) / Right(-)
        'cross_anterior': cross[1],     # Y: Anterior(+) / Posterior(-)
        'cross_superior': cross[2],     # Z: Superior(+) / Inferior(-)
        
        # Spatial displacement
        'lateral_displacement': displacement[0],    # X movement
        'anterior_displacement': displacement[1],   # Y movement  
        'apical_displacement': displacement[2],     # Z movement (negative = toward apex)
        
        # Derived features
        'is_leftward': cross[0] > 0.2,              # Curves to left (LCX)
        'is_forward': displacement[1] > 0,          # Goes anterior (LAD)
        'is_descending': displacement[2] < -20,     # Goes to apex (LAD)
        'is_horizontal': abs(displacement[2]) < 30, # Stays horizontal (LCX)
        
        # Traditional angle
        'dot_product': np.dot(main_norm, tree_norm),
        'angle_degrees': np.degrees(np.arccos(np.clip(np.dot(main_norm, tree_norm), -1.0, 1.0)))
    }


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
    # The first branch point should be the LAD/LCX bifurcation
    _, lmca_bifurcation_idx = vector_between_startpoint_and_closest_breakpoint(
        point_data['start_point'],
        distance_between_point_and_set(point_data['start_point'], point_data['all_branch_points']),
        point_data['all_branch_points'])
    lmca_bifurcation = point_data['all_branch_points'][lmca_bifurcation_idx]
    
    # Find where this tree reaches the bifurcation
    lmca_end_idx = None
    for i, point in enumerate(tree):
        if np.linalg.norm(point - lmca_bifurcation) < 1.0:
            lmca_end_idx = i
            break

    # Use the post-LMCA portion for analysis
    tree_post_lmca = tree[lmca_end_idx:]
    effective_start = tree[lmca_end_idx]
    tree_end = tree[-1]
    
    distances_to_lmca = np.linalg.norm(
    point_data['all_branch_points'] - lmca_bifurcation, 
    axis=1
    )
    mask = distances_to_lmca > 1.0  # Keep points more than 1mm away

    bp_without_lmca_bifurcation = point_data['all_branch_points'][mask]
    
    # =========================================================================
    # STEP 1: EXTRACT KEY ANATOMICAL FEATURES (UPDATED)
    # =========================================================================
    
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
    main_trunk_vector = point_data['start_point']-effective_start
    main_trunk_vector_norm = main_trunk_vector / np.linalg.norm(main_trunk_vector)

    # 2.2 This tree's direction - FROM bifurcation, not from aorta
    tree_direction_vector = tree[lmca_end_idx+10] - effective_start
    tree_direction_vector_norm = tree_direction_vector / np.linalg.norm(tree_direction_vector)

    # 2.3 Analyze direction using cross product
    direction_features = analyze_vessel_direction(
        main_trunk_vector_norm, 
        tree_direction_vector_norm,
        tree_end,
        effective_start
    )
    
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
    elif main_tree == 'LCX':
        primary_territory = LCX_TERRITORY
        secondary_territory = LAD_TERRITORY
        main_end_segments = LCX_END_SEGMENTS
    else:
        is_ramus = True
        primary_territory = LAD_TERRITORY
        secondary_territory = LCX_TERRITORY
        main_end_segments = LAD_END_SEGMENTS
    neutral_end = {13}    
    neutral_territory = {4}  # Shared segment
    # 3.1 Calculate territory coverage scores
    primary_score = len(segments_traversed.intersection((primary_territory.union(neutral_territory))))
    secondary_score = len(segments_traversed.intersection(secondary_territory))
    territory_ratio = primary_score / (primary_score + secondary_score + 1e-6)
    
    # =========================================================================
    # STEP 4: RULE-BASED CLASSIFICATION LOGIC (UPDATED)
    # =========================================================================
    
    is_main_trunk = (
        (total_length > 105 if main_tree=='LAD' else total_length > 70) and  # Long vessel (excluding LMCA)
        len(bp_without_lmca_bifurcation) >= 1 and  # Has multiple side branches
        (end_segment in main_end_segments.union(neutral_end)) and
        territory_ratio > 0.55 )
    
    if is_main_trunk:
        return main_tree  
    
    divergence_point = None
    divergence_idx = None
    max_distance_from_lmca = 0.0
    
    for i, point in enumerate(tree_post_lmca):
        for bp in bp_without_lmca_bifurcation:  # Skip LMCA bifurcation
            if np.linalg.norm(point - bp) < 3.0:
                # Calculate distance from this branch point to LMCA bifurcation
                dist_to_lmca = np.linalg.norm(bp - lmca_bifurcation)
                
                # Keep the branch point that is FURTHEST from LMCA
                if dist_to_lmca > max_distance_from_lmca:
                    divergence_point = point
                    divergence_idx = i
                    max_distance_from_lmca = dist_to_lmca
    # 4.2 CHECK IF THIS IS A DIAGONAL (LAD branch)
    if main_tree == 'LAD':
                # Find WHERE along the path this tree diverges (if at all)  
        if divergence_idx is not None and divergence_idx > 10 and divergence_idx < len(tree_post_lmca) - 10:
            # Vector along parent vessel (LAD/LCX) - going toward LMCA
            parent_vessel_vector = tree_post_lmca[divergence_idx - 10] - tree_post_lmca[divergence_idx]
            
            # Vector along branch (Diagonal/Marginal) - continuing away
            branch_vector = tree_post_lmca[divergence_idx + 10] - tree_post_lmca[divergence_idx]
            
            # Analyze the branch direction relative to parent vessel
            directional_features_branch = analyze_vessel_direction(
                parent_vessel_vector,  # Parent vessel direction (back toward LMCA)
                branch_vector,         # Branch direction (continuing forward)
                tree_post_lmca[-1],    # Branch endpoint
                tree_post_lmca[divergence_idx]  # Branch start (divergence point)
            )
        else:
            # Fallback if divergence is too close to start/end
            directional_features_branch = direction_features
                    
        # Calculate path length from bifurcation to divergence
        if divergence_idx is not None:
            path_to_divergence = np.sum(
                np.linalg.norm(np.diff(tree_post_lmca[:divergence_idx], axis=0), axis=1)
            )
        else:
            path_to_divergence = total_length  # No divergence = probably main trunk

        # Branches should diverge EARLY after bifurcation
        is_diagonal = (
            #20 < total_length < 140 and
            divergence_point is not None and
            directional_features_branch['angle_degrees'] > 100 and  # Branches at moderate angle
            lateral_displacement > 10 and
            primary_score > 0 and
            end_segment in {4,10, 11, 13, 14, 15, 16, 17})
        
        if is_diagonal:
            return f'Diagonal_temp'
        
        is_septal = (
        directional_features_branch['angle_degrees'] > 70 and  # Sharp angle
        end_segment in {2, 3, 8, 9, 14} and
        directional_features_branch['anterior_displacement'] < 0
        )
        if is_septal:
            return 'Septal'
    
    # 4.3 CHECK IF THIS IS AN OBTUSE MARGINAL (LCX branch)
    if main_tree == 'LCX':
        if divergence_idx is not None and divergence_idx > 10 and divergence_idx < len(tree_post_lmca) - 10:
            # Vector along parent vessel (LAD/LCX) - going toward LMCA
            parent_vessel_vector = tree_post_lmca[divergence_idx - 10] - tree_post_lmca[divergence_idx]
            parent_vessel_vector_norm = parent_vessel_vector / np.linalg.norm(parent_vessel_vector)
            
            # Vector along branch (Diagonal/Marginal) - continuing away
            branch_vector = tree_post_lmca[divergence_idx + 10] - tree_post_lmca[divergence_idx]
            branch_vector_norm = branch_vector / np.linalg.norm(branch_vector)
            
            # Analyze the branch direction relative to parent vessel
            directional_features_branch = analyze_vessel_direction(
                parent_vessel_vector_norm,  # Parent vessel direction (back toward LMCA)
                branch_vector_norm,         # Branch direction (continuing forward)
                tree_post_lmca[-1],    # Branch endpoint
                tree_post_lmca[divergence_idx]  # Branch start (divergence point)
            )
                        
        is_marginal = (
            #(20 < directional_features_branch['angle_degrees'] < 120 if (divergence_idx>10) else True) and  # Branches at wider angle
            primary_score > 0 and  # Passes through LCX territory
            end_segment in {5, 6, 7, 11, 12, 16} )  # Ends in lateral segments
        
        if is_marginal:
            return f'Marginal_temp'
 
    if main_tree not in ['LAD', 'LCX']:
        is_ramus = (
            total_length > 30 and
            direction_features['angle_degrees'] < 20
        )
        if is_ramus:
            return 'Ramus Intermedius'
    
    # If none of the above, classify as outlier or small branch
    if total_length < 30:
        return 'Small Branch'
    else:
        return 'Outlier'

def wrap_label_artery_segment(tree_labels, trees, points):
    """
    Post-process tree labels to ensure only the longest tree is labeled as LAD or LCX.
    If multiple trees are labeled as the same main vessel, keep the longest one and 
    reclassify the others as diagonals (for LAD) or marginals (for LCX).
    Also renumber all diagonals and marginals based on their distance from LMCA bifurcation.
    
    Parameters:
    -----------
    tree_labels : dict
        Dictionary of tree labels from initial classification
    trees : list of np.ndarray
        All traced trees
    points : dict
        Point data containing branch points and other info
    main_trunk : dict
        Dictionary mapping tree indices to their main vessel ('LAD' or 'LCX')
        
    Returns:
    --------
    dict : Corrected tree labels
    """
    global diagonal_count, marginal_count
    
    # Get LMCA bifurcation
    _, lmca_bifurcation_idx = vector_between_startpoint_and_closest_breakpoint(
        points['start_point'],
        distance_between_point_and_set(points['start_point'], points['all_branch_points']),
        points['all_branch_points'])
    lmca_bifurcation = points['all_branch_points'][lmca_bifurcation_idx]
    
    # Find all trees labeled as LAD
    lad_trees = {idx: tree for idx, tree in zip(tree_labels.keys(), trees) 
                 if tree_labels[idx] == 'LAD'}
    
    # Find all trees labeled as LCX
    lcx_trees = {idx: tree for idx, tree in zip(tree_labels.keys(), trees) 
                 if tree_labels[idx] == 'LCX'}
    
    # =========================================================================
    # Handle multiple LADs
    # =========================================================================
    if len(lad_trees) > 1:
        print(f"\nWarning: Found {len(lad_trees)} trees labeled as LAD")
        
        # Calculate lengths for each LAD tree
        lad_lengths = {}
        for idx, tree in lad_trees.items():
            length = np.sum(np.linalg.norm(np.diff(tree, axis=0), axis=1))
            lad_lengths[idx] = length
            print(f"  Tree {idx}: {length:.1f} mm")
        
        # Find the longest one
        longest_lad_idx = max(lad_lengths, key=lad_lengths.get)
        print(f"  Keeping Tree {longest_lad_idx} as LAD (longest)")
        
        # Reclassify the others as diagonals (will be renumbered later)
        for idx in lad_trees.keys():
            if idx != longest_lad_idx:
                tree_labels[idx] = 'Diagonal_temp'  # Temporary label
                print(f"  Reclassified Tree {idx} as diagonal (will be renumbered)")
    
    # =========================================================================
    # Handle multiple LCXs
    # =========================================================================
    if len(lcx_trees) > 1:
        print(f"\nWarning: Found {len(lcx_trees)} trees labeled as LCX")
        
        # Calculate lengths for each LCX tree
        lcx_lengths = {}
        for idx, tree in lcx_trees.items():
            length = np.sum(np.linalg.norm(np.diff(tree, axis=0), axis=1))
            lcx_lengths[idx] = length
            print(f"  Tree {idx}: {length:.1f} mm")
        
        # Find the longest one
        longest_lcx_idx = max(lcx_lengths, key=lcx_lengths.get)
        print(f"  Keeping Tree {longest_lcx_idx} as LCX (longest)")
        
        # Reclassify the others as marginals (will be renumbered later)
        for idx in lcx_trees.keys():
            if idx != longest_lcx_idx:
                tree_labels[idx] = 'Marginal_temp'  # Temporary label
                print(f"  Reclassified Tree {idx} as marginal (will be renumbered)")
    
    # =========================================================================
    # Renumber ALL diagonals based on distance from LMCA
    # =========================================================================
    
    # Find all diagonals (including those just reclassified)
    diagonal_indices = [idx for idx, label in tree_labels.items() 
                       if label.startswith('D') or label == 'Diagonal_temp']
    
    if diagonal_indices:
        print(f"\nRenumbering {len(diagonal_indices)} diagonal(s) based on distance from LMCA")
        
        # Calculate divergence distance from LMCA for each diagonal
        diagonal_distances = {}
        
        for idx in diagonal_indices:
            tree = trees[list(tree_labels.keys()).index(idx)]
            
            # Skip LMCA portion
            lmca_end_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 3.0:
                    lmca_end_idx = i
                    break
            
            if lmca_end_idx is None:
                lmca_end_idx = len(tree) // 10
            
            tree_post_lmca = tree[lmca_end_idx:]
            
            # Find divergence point (furthest branch point from LMCA along this tree)
            max_distance_from_lmca = 0.0
            divergence_point = None
            
            bp_without_lmca = points['all_branch_points'][
                np.linalg.norm(points['all_branch_points'] - lmca_bifurcation, axis=1) > 1.0
            ]
            
            for point in tree_post_lmca:
                for bp in bp_without_lmca:
                    if np.linalg.norm(point - bp) < 3.0:
                        dist_to_lmca = np.linalg.norm(bp - lmca_bifurcation)
                        if dist_to_lmca > max_distance_from_lmca:
                            max_distance_from_lmca = dist_to_lmca
                            divergence_point = bp
            
            # If no divergence found, use distance along tree
            if divergence_point is None:
                max_distance_from_lmca = np.linalg.norm(tree_post_lmca[0] - lmca_bifurcation)
            
            diagonal_distances[idx] = max_distance_from_lmca
            print(f"  Tree {idx}: Divergence distance = {max_distance_from_lmca:.1f} mm from LMCA")
        
        # Sort diagonals by distance (closest first = D1, furthest = DN)
        sorted_diagonals = sorted(diagonal_distances.items(), key=lambda x: x[1])
        
        # Renumber
        for new_number, (idx, distance) in enumerate(sorted_diagonals, start=1):
            tree_labels[idx] = f'D{new_number}'
            print(f"  Tree {idx} renumbered as D{new_number}")
        
        # Update global count
        diagonal_count = len(diagonal_indices)
    
    # =========================================================================
    # Renumber ALL marginals based on distance from LMCA
    # =========================================================================
    
    # Find all marginals (including those just reclassified)
    marginal_indices = [idx for idx, label in tree_labels.items() 
                       if label.startswith('OM') or label == 'Marginal_temp']
    
    if marginal_indices:
        print(f"\nRenumbering {len(marginal_indices)} marginal(s) based on distance from LMCA")
        
        # Calculate divergence distance from LMCA for each marginal
        marginal_distances = {}
        
        for idx in marginal_indices:
            tree = trees[list(tree_labels.keys()).index(idx)]
            
            # Skip LMCA portion
            lmca_end_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 3.0:
                    lmca_end_idx = i
                    break
            
            if lmca_end_idx is None:
                lmca_end_idx = len(tree) // 10
            
            tree_post_lmca = tree[lmca_end_idx:]
            
            # Find divergence point (furthest branch point from LMCA along this tree)
            max_distance_from_lmca = 0.0
            divergence_point = None
            
            bp_without_lmca = points['all_branch_points'][
                np.linalg.norm(points['all_branch_points'] - lmca_bifurcation, axis=1) > 1.0
            ]
            
            for point in tree_post_lmca:
                for bp in bp_without_lmca:
                    if np.linalg.norm(point - bp) < 3.0:
                        dist_to_lmca = np.linalg.norm(bp - lmca_bifurcation)
                        if dist_to_lmca > max_distance_from_lmca:
                            max_distance_from_lmca = dist_to_lmca
                            divergence_point = bp
            
            # If no divergence found, use distance along tree
            if divergence_point is None:
                max_distance_from_lmca = np.linalg.norm(tree_post_lmca[0] - lmca_bifurcation)
            
            marginal_distances[idx] = max_distance_from_lmca
            print(f"  Tree {idx}: Divergence distance = {max_distance_from_lmca:.1f} mm from LMCA")
        
        # Sort marginals by distance (closest first = OM1, furthest = OMN)
        sorted_marginals = sorted(marginal_distances.items(), key=lambda x: x[1])
        
        # Renumber
        for new_number, (idx, distance) in enumerate(sorted_marginals, start=1):
            tree_labels[idx] = f'OM{new_number}'
            print(f"  Tree {idx} renumbered as OM{new_number}")
        
        # Update global count
        marginal_count = len(marginal_indices)
    
    # =========================================================================
    # Optional: Check for missing main vessels
    # =========================================================================
    has_lad = 'LAD' in tree_labels.values()
    has_lcx = 'LCX' in tree_labels.values()
    
    if not has_lad:
        print("\nWarning: No tree labeled as LAD")
    if not has_lcx:
        print("\nWarning: No tree labeled as LCX")
    
    return tree_labels

def create_labeled_segments(trees, tree_labels, point_data):
    """
    Create labeled coronary artery segments based on anatomical structure.
    Ensures branches diverge from main vessels (LAD/LCX) not from other branches.
    
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
    # 1. CREATE LMCA SEGMENT
    # =========================================================================
    distances_to_start = np.linalg.norm(
        point_data['all_branch_points'] - point_data['start_point'], 
        axis=1
    )
    lmca_bifurcation_idx = np.argmin(distances_to_start)
    lmca_bifurcation = point_data['all_branch_points'][lmca_bifurcation_idx]
    
    bp_without_lmca = np.delete(point_data['all_branch_points'], lmca_bifurcation_idx, axis=0)
    
    # Use any tree to get the LMCA segment (they're all the same at the start)
    first_tree = trees[0]
    lmca_segment = []
    
    for i, point in enumerate(first_tree):
        lmca_segment.append(point)
        if np.linalg.norm(point - lmca_bifurcation) < 1.5:
            break
    
    labeled_segments['LMCA'] = np.array(lmca_segment)
    
    # =========================================================================
    # 2. FIRST PASS: CREATE LAD AND LCX SEGMENTS
    # =========================================================================
    lad_segment = None
    lcx_segment = None
    
    for tree_idx, (tree, label) in enumerate(zip(trees, tree_labels.values())):
        if label == 'LAD':
            # Find where this tree reaches the LMCA bifurcation
            bifurcation_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 1.5:
                    bifurcation_idx = i
                    break
            
            if bifurcation_idx is None: 
                bifurcation_idx = len(tree) // 10
            
            lad_segment = tree[bifurcation_idx:]
            labeled_segments['LAD'] = lad_segment
            
        elif label == 'LCX':
            # Find where this tree reaches the LMCA bifurcation
            bifurcation_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 1.5:
                    bifurcation_idx = i
                    break
            
            lcx_segment = tree[bifurcation_idx:]
            labeled_segments['LCX'] = lcx_segment
    
    # =========================================================================
    # 3. SECOND PASS: CREATE BRANCH SEGMENTS (DIAGONALS AND MARGINALS)
    # =========================================================================
    
    for tree_idx, (tree, label) in enumerate(zip(trees, tree_labels.values())):
        
        # Skip main vessels and outliers
        if label in ['LAD', 'LCX']:
            continue
            
        if label in ['Outlier', 'Small Branch']:
            bifurcation_idx = None
            for i, point in enumerate(tree):
                if np.linalg.norm(point - lmca_bifurcation) < 1.5:
                    bifurcation_idx = i
                    break
            if bifurcation_idx is not None:
                labeled_segments[label] = tree[bifurcation_idx:]
            continue
        
        # -----------------------------------------------------------------
        # For branches: Find divergence from LAD or LCX (not from other branches)
        # -----------------------------------------------------------------
        
        # Skip LMCA portion
        lmca_end_idx = None
        for i, point in enumerate(tree):
            if np.linalg.norm(point - lmca_bifurcation) < 1.5:
                lmca_end_idx = i
                break
        
        if lmca_end_idx is None:
            lmca_end_idx = len(tree) // 10
        
        tree_post_lmca = tree[lmca_end_idx:]
        
        # Determine which main vessel this branch comes from
        is_diagonal = label.startswith('D') or label == 'Diagonal_temp' or label == 'Septal'
        is_marginal = label.startswith('OM') or label == 'Marginal_temp'
        
        parent_vessel = lad_segment if is_diagonal else lcx_segment
        
        if parent_vessel is None:
            # Fallback if parent vessel not found
            print(f"Warning: Parent vessel not found for {label}, using fallback")
            labeled_segments[label] = tree_post_lmca
            continue
        
        # Find the divergence point where this branch leaves the parent vessel
        divergence_idx = None
        min_divergence_distance = float('inf')
        
        # For each point in the branch tree
        for i, branch_point in enumerate(tree_post_lmca):
            # Find the closest point on the parent vessel
            distances_to_parent = np.linalg.norm(parent_vessel - branch_point, axis=1)
            min_dist_to_parent = np.min(distances_to_parent)
            
            # If this point is close to the parent vessel
            if min_dist_to_parent < 3.0:
                # Check if there's a branch point nearby
                for bp in bp_without_lmca:
                    if np.linalg.norm(branch_point - bp) < 3.0:
                        # Calculate how far along the parent vessel this divergence is
                        closest_parent_idx = np.argmin(distances_to_parent)
                        path_length_on_parent = np.sum(
                            np.linalg.norm(np.diff(parent_vessel[:closest_parent_idx+1], axis=0), axis=1)
                        )
                        
                        # Keep the divergence point that is FURTHEST along the parent vessel
                        # This ensures we diverge from the main vessel, not from another branch
                        if path_length_on_parent < min_divergence_distance:
                            min_divergence_distance = path_length_on_parent
                            divergence_idx = lmca_end_idx + i
        
        # Alternative method: Find the last point where branch touches parent vessel
        if divergence_idx is None:
            print(f"Using alternative divergence detection for {label}")
            last_contact_idx = None
            
            for i, branch_point in enumerate(tree_post_lmca):
                distances_to_parent = np.linalg.norm(parent_vessel - branch_point, axis=1)
                min_dist_to_parent = np.min(distances_to_parent)
                
                # If this point is very close to parent vessel
                if min_dist_to_parent < 2.0:
                    last_contact_idx = i
            
            if last_contact_idx is not None:
                divergence_idx = lmca_end_idx + last_contact_idx
        
        # Use from divergence point to end
        if divergence_idx is not None:
            labeled_segments[label] = tree[divergence_idx:]
            print(f"  {label}: Diverges at index {divergence_idx}, segment length = {len(tree[divergence_idx:])} points")
        else:
            # Final fallback: use from after LMCA
            labeled_segments[label] = tree_post_lmca
            print(f"  Warning: Could not find divergence for {label}, using post-LMCA segment")
    
    # =========================================================================
    # 4. VALIDATION: Check that branches don't overlap significantly
    # =========================================================================
    
    branch_labels = [label for label in labeled_segments.keys() 
                     if label.startswith('D') or label.startswith('OM')]
    
    print(f"\nValidating {len(branch_labels)} branch segments...")
    
    for i, label1 in enumerate(branch_labels):
        for label2 in branch_labels[i+1:]:
            seg1 = labeled_segments[label1]
            seg2 = labeled_segments[label2]
            
            # Check for overlap (points within 2mm of each other)
            overlap_count = 0
            for p1 in seg1[::2]:  # Sample every 2nd point for efficiency
                min_dist = np.min(np.linalg.norm(seg2 - p1, axis=1))
                if min_dist < 2.0:
                    overlap_count += 1
            
            overlap_ratio = overlap_count / (len(seg1[::2]) + 1)
            
            if overlap_ratio > 0.3:  # More than 30% overlap
                print(f"  Warning: Significant overlap between {label1} and {label2} ({overlap_ratio*100:.1f}%)")
    
    return labeled_segments

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
    #id_list = ["179","224","333","377","560","603","708","714","770","865"]
    id = '770'
    series_id = '0020'
    Parent_dir = f'assets/imagecas/nii_images_sample_tracing/{id}.img'
    point_path = Parent_dir + '/tracer_points.json'
    path_main_tree = Parent_dir + '/tree_angles.json'
    path_trees = Parent_dir + '/path_tracing/combined_paths'
    path_combined_tree = Parent_dir + f'/path_tracing/combined_paths/{id}.img_combined_tree_spline.vtk'

    points = load_points_from_json(point_path)
    trees = load_vtk_trees_without_spline(path_trees)
    main_trunk = json.load(open(path_main_tree, 'r'))['tree_labels']
    
    # AHA 17-Segment Model Territories
    LAD_TERRITORY = {10,13,14,15,16,17}    # Anterior, Anteroseptal, Apex
    LCX_TERRITORY = {1,2,5,6,7,11,12}              # Anterolateral, Inferolateral
 
    # Define key segments for termination
    LAD_END_SEGMENTS = {15, 17, 14, 1, 7}         # Apex, Apical Anterior/Anteroseptal
    LCX_END_SEGMENTS = {11, 12, 16, 5, 6, 1}     # Mid/Apical Lateral

    diagonal_count = 0
    marginal_count = 0
    tree_data = {}
    end_segments_data = {}
    tree_labels = {}
    for idx, (end_point, tree) in zip(main_trunk.keys(), zip(points['end_points'], trees)):
        tree_idx_str = idx
        
        tree_label = label_artery_segment_given_main_branch(tree, points, main_trunk[idx],all_trees=trees)
        tree_labels[tree_idx_str] = tree_label
        print(f"Tree {tree_idx_str} labeled as {tree_label}")
    print("Classified Tree Labels:", tree_labels)
    fixed_tree_labels = wrap_label_artery_segment(tree_labels, trees, points)
    
    labeled_segments = create_labeled_segments(trees, fixed_tree_labels, points)
    output_folder = Parent_dir + '/labeled_segments_vtk'
    save_labeled_segments_to_vtk(labeled_segments, output_folder)