import numpy as np
import json
import glob
import os
import vtk
from utils import vtk_to_numpy



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

def get_closest_segment_key_to_tree_point(tree_points, atlas):
    closest_segments = set()
    # make step size 5
    for point in tree_points[::1]:
        com_coordinatres = np.array([value for value in atlas.values()])
        distances = np.linalg.norm(com_coordinatres - point, axis=1)
        closest_index = np.argmin(distances)
        closest_segments.add(list(atlas.keys())[closest_index])
    return closest_segments

def get_closest_segment_to_end_point(end_point, atlas):
    com_coordinatres = np.array([value for value in atlas.values()])
    distances = np.linalg.norm(com_coordinatres - end_point, axis=1)
    closest_index = np.argmin(distances)
    return list(atlas.keys())[closest_index]

# AHA 17-Segment Model Territories
LAD_TERRITORY = {4,10,11,13,14,15,16,17}  # Anterior, Anteroseptal, Apex
LCX_TERRITORY = {1,2,5,6,7,12}       # Anterolateral, Inferolateral
#RCA_TERRITORY = {3, 4, 9, 10, 15}       # Inferior, Inferoseptal (useful for context)

# Define key segments for termination
LAD_END_SEGMENTS = {17, 13, 14} # Apex, Apical Anterior/Anteroseptal
LCX_END_SEGMENTS = {11, 12, 16, 5} # Mid/Apical Lateral



def classify_arteries(tree_data, end_segments_data):
    """
    Classifies coronary artery trees based on their path and termination segment.

    Args:
        tree_data (dict): {tree_idx: {segment_keys}} e.g., {'1': {'segment_com_1', 'segment_com_7', ...}}
        end_segments_data (dict): {tree_idx: end_segment_key} e.g., {'1': 'segment_com_17'}

    Returns:
        dict: {tree_idx: 'Label'} e.g., {'1': 'LAD', '2': 'LCX', ...}
    """
    artery_scores = {}
    artery_labels = {}

    for tree_idx, segment_keys in tree_data.items():
        # Convert segment keys (e.g., 'segment_com_17') to integer numbers
        segments_passed = {int(key.split('_')[-1]) for key in segment_keys}
        
        # Calculate how many segments in the path belong to each territory
        lad_score = len(segments_passed.intersection(LAD_TERRITORY))
        lcx_score = len(segments_passed.intersection(LCX_TERRITORY))
        
        # Get the termination segment number
        end_segment_key = end_segments_data[tree_idx]
        end_segment = int(end_segment_key.split('_')[-1])

        artery_scores[tree_idx] = {
            'lad_score': lad_score,
            'lcx_score': lcx_score,
            'path_length': len(segments_passed), # Use number of segments as a proxy for length
            'end_segment': end_segment
        }

    # Find best candidate for LAD
    best_lad_candidate = max(
        artery_scores.keys(), 
        key=lambda k: (artery_scores[k]['lad_score'], artery_scores[k]['path_length'])
    )
    
    # Find best candidate for LCX
    best_lcx_candidate = max(
        artery_scores.keys(), 
        key=lambda k: (artery_scores[k]['lcx_score'], artery_scores[k]['path_length'])
    )

    # Assign labels to main trunks
    artery_labels[best_lad_candidate] = 'LAD'
    if best_lcx_candidate != best_lad_candidate:
        artery_labels[best_lcx_candidate] = 'LCX'

    # 4. Classify the Remaining Branches (Diagonals, Marginals)
    for tree_idx, scores in artery_scores.items():
        if tree_idx in artery_labels:
            continue # Skip already-labeled main trunks

        # Rule for Diagonals: Branches that are in LAD territory
        if scores['lad_score'] > scores['lcx_score'] and scores['end_segment'] in LAD_TERRITORY:
            artery_labels[tree_idx] = 'Diagonal'
        
        # Rule for Obtuse Marginals: Branches that are in LCX territory
        elif scores['lcx_score'] > scores['lad_score'] and scores['end_segment'] in LCX_TERRITORY:
            artery_labels[tree_idx] = 'Obtuse Marginal'
            
        # Fallback for ambiguous branches
        else:
            artery_labels[tree_idx] = 'Unknown Branch'
            
    return artery_labels


if __name__ == "__main__":
    id = '0010'
    series_id = '0020'
    path = 'output.json'
    path_atlas = f'assets/data/{id}/processed/misc/atlas.json'
    path_trees = f'/Users/jacob/OneDrive/Uni/9. Semester/SpecialProjekt/sp/assets/data/CoronaryTracing/CFA-PILOT_{id}_SERIES{series_id}/path_tracing/combined_paths'
    points = load_points_from_json(path)
    atlas = load_points_from_json(path_atlas)
    trees = load_vtk_trees_without_spline(path_trees)

    tree_data = {}
    end_segments_data = {}
    for idx, (end_point, tree) in enumerate(zip(points['end_points'], trees)):
        tree_idx_str = str(idx + 1)
        
        # Get the set of segments the tree passes through
        closest_com_segments = get_closest_segment_key_to_tree_point(tree, atlas['points'][()])
        tree_data[tree_idx_str] = closest_com_segments
        
        # Get the segment where the tree terminates
        closest_end_segment = get_closest_segment_to_end_point(end_point, atlas['points'][()])
        end_segments_data[tree_idx_str] = closest_end_segment

    # Now, classify the arteries
    classified_arteries = classify_arteries(tree_data, end_segments_data)
    
    print("Artery Classification Results:")
    for tree_idx, label in classified_arteries.items():
        print(f"  Tree {tree_idx}: {label}")