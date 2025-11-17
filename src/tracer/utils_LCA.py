from pathlib import Path
import glob
import json
import re

import numpy as np
from tqdm import tqdm

from IV_generate_segments import wrap_lv_segments

def convert_txt_to_vtk(input_txt_path, output_vtk_path):
    # Read the points from the text file
    with open(input_txt_path, 'r') as txt_file:
        lines = txt_file.readlines()

    # Open the output .vtk file
    with open(output_vtk_path, 'w') as vtk_file:
        # Write the VTK header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Point data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write the points
        vtk_file.write(f"POINTS {len(lines)} float\n")
        for line in lines:
            x, y, z = map(float, line.split())
            vtk_file.write(f"{x} {y} {z}\n")

        # Write the vertices (optional, for visualization as points)
        vtk_file.write(f"VERTICES {len(lines)} {len(lines) * 2}\n")
        for i in range(len(lines)):
            vtk_file.write(f"1 {i}\n")

class TracerPointExtractor:
    def __init__(self,
                 tracer_folder_path : str = "assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036"):

        self.tracer_folder_path = tracer_folder_path
        self.combined_paths_folder = Path(self.tracer_folder_path) / "path_tracing" / "combined_paths"
        self.individual_paths_folder = Path(self.tracer_folder_path) / "path_tracing" / "tracing_with_momentum_individual_paths"
        self.total_trees = self._get_amount_of_traced_paths()
        self.tree_files = self._get_traced_path_files()


        self.end_points = self._get_end_point_from_vtk()


        self.start_point = self._parse_xyz_coordinates(
                    self._get_start_point_from_vtk()
        )

        self.all_bp = self._get_all_bp_points()

    def _get_amount_of_traced_paths(self):


        return len([p for p in self.combined_paths_folder.glob("*_traced_path_*.vtk")
            if not p.name.endswith("spline.vtk")])

    def _get_traced_path_files(self):


        return [p for p in self.combined_paths_folder.glob("*_traced_path_*.vtk")
            if not p.name.endswith("spline.vtk")]

    def _get_end_point_from_vtk(self):
        last_points = []

        for vtk_file_path in self.tree_files:
            with open(vtk_file_path, 'r') as vtk_file:
                lines = vtk_file.readlines()

            # Find the POINTS section
            points_start = None
            num_points = 0
            for i, line in enumerate(lines):
                if line.startswith("POINTS"):
                    points_start = i + 1
                    num_points = int(line.split()[1])  # Extract the number of points
                    break

            if points_start is None:
                raise ValueError("No POINTS section found in the .vtk file.")

            # Extract the points
            points = []
            for line in lines[points_start:]:
                if line.startswith("LINES"):  # Stop parsing when the LINES section starts
                    break
                try:
                    points.append(list(map(float, line.split())))
                except ValueError:
                    continue  # Skip lines that cannot be converted to floats
            if len(points[-1][:3]) == 0:
                last_points.append(points[-2][:3] if points else None)
            else:
                last_points.append(points[-1][:3] if points else None)

        return last_points

    def _get_start_point_from_vtk(self):
        start_point_files = glob.glob(str(self.individual_paths_folder / "*start_point.txt"))


        with open(start_point_files[0], 'r') as vtk_file:
            return vtk_file.read().strip()  # Read and store the content

    def _get_all_bp_points(self, distance_threshold=0.4, prune_end_n=0):
        # bp_point_files = glob.glob(str(self.individual_paths_folder / "*all_bp.txt"))



        # with open(bp_point_files[0], 'r') as vtk_file:
        #     return vtk_file.read().strip()  # Read and store the content

        detected_bp = branch_point_detector_backward_tracking(output_path, distance_threshold=0.55, prune_end_n=10, cluster_eps_multiplier=3.0, min_cluster_size=10, tolerance=0.5)

        valid_bp, metadata = validate_branch_points_divergence(
            output_path,
            detected_bp,
            distance_threshold=0.6,
            trajectory_length=30,
            min_angle_threshold=15.0,
            min_diverging_paths=2
        )

        return valid_bp


    def _parse_xyz_coordinates(self, coordinate_string):

        # Split the string into lines
        lines = coordinate_string.strip().split("\n")

        # Convert each line into a list of floats
        coordinates = [list(map(float, line.split())) for line in lines]

        return coordinates

    def extract_tracer_points(self, atlas_json_path=None):
        atlas_points = self.load_atlas_json(atlas_json_path) if atlas_json_path else {}

        return {
            "start_point": self.start_point,
            "end_points": self.end_points,
            "all_branch_points": self.all_bp,
            "atlas_points": atlas_points
        }

    @staticmethod
    def load_atlas_json(atlas_json_path):
        with open(atlas_json_path, 'r') as f:
            atlas_data = json.load(f)
        return list(atlas_data["points"].values())

    def return_tracer_points_as_json(self, atlas_json_path=None, output_path=None):

        tracer_points = self.extract_tracer_points(atlas_json_path=atlas_json_path)

        with open(output_path, "w") as json_file:
            json.dump(tracer_points, json_file, indent=4)
        return json.dumps(tracer_points, indent=4)

def json_to_vtk_points(json_file_path, output_folder=".", prefix="points"):
    """
    Converts JSON coordinate data to VTK point files for visualization in Slicer.

    Args:
        json_file_path (str): Path to the input JSON file
        output_folder (str): Directory to save the VTK files
        prefix (str): Prefix for the output VTK files

    Returns:
        list: Paths to the created VTK files
    """
    from pathlib import Path

    # Read JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    created_files = []

    # Process each set of points
    for key, points in data.items():
        output_file = output_folder / f"{prefix}_{key}.vtk"

        # Flatten the points if needed (for start_point which is nested)
        if len(points) > 0 and isinstance(points[0], list):
            flat_points = points
        else:
            flat_points = [points]

        # Write VTK file
        with open(output_file, 'w') as vtk_file:
            # Write VTK header
            vtk_file.write("# vtk DataFile Version 3.0\n")
            vtk_file.write(f"{key} data\n")
            vtk_file.write("ASCII\n")
            vtk_file.write("DATASET POLYDATA\n")

            # Write points
            vtk_file.write(f"POINTS {len(flat_points)} float\n")
            for point in flat_points:
                vtk_file.write(f"{point[0]} {point[1]} {point[2]}\n")

            # Write vertices (for visualization as points)
            vtk_file.write(f"\nVERTICES {len(flat_points)} {len(flat_points) * 2}\n")
            for i in range(len(flat_points)):
                vtk_file.write(f"1 {i}\n")

        created_files.append(str(output_file))
        print(f"Created: {output_file}")

    return created_files

def compute_distances_from_root(data):
    """
    Computes the Euclidean distance from each point to the start_point (root).

    Args:
        data (dict): Dictionary with 'start_point', 'end_points', and 'all_branch_points'

    Returns:
        dict: Dictionary with the same structure, but each point is now [x, y, z, distance]
    """
    # Extract the start point (root)
    start_point = np.array(data['start_point'][0])

    result = {}

    for key, points in data.items():
        points_with_distance = []

        for point in points:
            point_array = np.array(point)
            # Compute Euclidean distance
            distance = np.linalg.norm(point_array - start_point)
            # Append distance as the 4th element
            point_with_dist = point + [distance]
            points_with_distance.append(point_with_dist)

        result[key] = points_with_distance

    return result

def save_distances_to_json(input_json_path, output_json_path):
    """
    Reads JSON, computes distances, and saves to a new JSON file.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    result = compute_distances_from_root(data)

    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Distances computed and saved to {output_json_path}")
    return result

def vectors_to_root(type_points, load_path='output.json'):
    # Load the data
    with open(load_path, 'r') as f:
        data = json.load(f)

    start_point = np.array(data['start_point'][0])
    branch_points = np.array(data[type_points])

    # Compute vectors from each branch point to root
    vectors = start_point - branch_points

    # Compute norms (magnitudes)
    norms = np.linalg.norm(vectors, axis=1)

    # Normalize vectors for direction
    normalized_vectors = vectors / norms[:, np.newaxis]

    # Create output for Slicer visualization
    output = {
        "start_point": data['start_point'][0],
        "branch_points": data['all_branch_points'],
        "vectors_to_root": vectors.tolist(),
        "vector_norms": norms.tolist(),
        "normalized_vectors": normalized_vectors.tolist()
    }

    # Save results
    with open('vector_analysis.json', 'w') as f:
        json.dump(output, f, indent=4)

    # Print summary
    for i, (bp, norm) in enumerate(zip(branch_points, norms)):
        print(f"Branch point {i}: {bp} -> Distance to root: {norm:.2f}")

def vec_from_point_to_point(point_a, point_b):
    return np.array(point_b) - np.array(point_a)

def visualize_slicer_vecs(load_path='output_with_distances.json',
                          output_path='branch_to_root_lines.mrk.json',
                          type_points='all_branch_points',
                          vtk_load_path=None):

    with open(load_path, 'r') as f:
        data = json.load(f)



    #start_point = np.array(data['start_point'][0])
    start_point = np.array(data["all_branch_points"][min(range(len(data["all_branch_points"])), key=lambda i: data["all_branch_points"][i][-1])][:3])
    branch_points = vtk_to_numpy(vtk_load_path)

    # Create Slicer markup JSON format
    markup = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Line",
            "coordinateSystem": "LPS",
            "controlPoints": []
        }]
    }

    # Add line from each branch point to root
    lines = []
    for i, bp in enumerate(branch_points):
        norm = np.linalg.norm(start_point - bp)
        line = {
            "type": "Line",
            "coordinateSystem": "LPS",
            "locked": False,
            "controlPoints": [
                {"position": bp.tolist()},
                {"position": start_point.tolist()}
            ]
        }
        lines.append(line)

    markup["markups"] = lines

    with open(output_path, 'w') as f:
        json.dump(markup, f, indent=2)

    print(f"Created {output_path} - load this in Slicer")

def vtk_to_numpy(vtk_file_path):
    """
    Reads all points from a .vtk file and converts them into a NumPy array.

    Args:
        vtk_file_path (str): Path to the .vtk file.

    Returns:
        np.ndarray: A NumPy array containing the points.
    """
    with open(vtk_file_path, 'r') as vtk_file:
        lines = vtk_file.readlines()

    # Find the POINTS section
    points_start = None
    num_points = 0
    for i, line in enumerate(lines):
        if line.startswith("POINTS"):
            points_start = i + 1
            num_points = int(line.split()[1])  # Extract the number of points
            break

    if points_start is None:
        raise ValueError("No POINTS section found in the .vtk file.")

    # Extract the points - flatten all values first
    flat_values = []
    for line in lines[points_start:]:
        if line.startswith("LINES"):  # Stop parsing when the LINES section starts
            break
        try:
            flat_values.extend(list(map(float, line.split())))
        except ValueError:
            continue  # Skip lines that cannot be converted to floats

    # Check if the number of values is divisible by 3
    if len(flat_values) % 3 != 0:
        raise ValueError(f"Number of values ({len(flat_values)}) is not divisible by 3.")

    # Split into chunks of 3 (x, y, z coordinates)
    points = [flat_values[i:i+3] for i in range(0, len(flat_values), 3)]

    return np.array(points)

def numpy_to_vtk(points_array, output_vtk_path):
    """
    Writes a NumPy array of points to a .vtk file.

    Args:
        points_array (np.ndarray): A NumPy array of shape (N, 3) containing the points.
        output_vtk_path (str): Path to the output .vtk file.
    """
    with open(output_vtk_path, 'w') as vtk_file:
        # Write VTK header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Point data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")

        # Write points
        num_points = points_array.shape[0]
        vtk_file.write(f"POINTS {num_points} float\n")
        for point in points_array:
            vtk_file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # Write vertices (for visualization as points)
        vtk_file.write(f"\nVERTICES {num_points} {num_points * 2}\n")
        for i in range(num_points):
            vtk_file.write(f"1 {i}\n")

    print(f"Created: {output_vtk_path}")

def tree_run_through_points(tree_path, json_path='output.json'):



    points = vtk_to_numpy(tree_path)
    threshold = 1.5



    with open(json_path, 'r') as f:
        data = json.load(f)

    all_branch_points = np.array(data['all_branch_points'])


    # Initialize with infinity for each combined point
    min_distances = np.full(len(all_branch_points), np.inf)

    for i, point in enumerate(points):
        # Calculate distances from current point to all combined points
        distances = np.linalg.norm(all_branch_points - point, axis=1)
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances)

    min_distances < threshold
    return min_distances, min_distances < threshold

def distance_between_point_and_set(point, point_set):
    '''Calculate distances between a single point and a set of points.'''
    distances = np.linalg.norm(point_set - point, axis=1)
    try:
        return np.argsort(distances)  # Return index of the closest point (excluding itself)
    except IndexError:
        return None

def classify_trees_pipeline(
        folder_path = 'assets/data/IMGCAS_tracing/182.img',
        segmentation_path = 'assets/data/IMGCAS_tracing/182.img/bartholinator/182.img_pred.nii.gz',
        image_path = 'assets/data/IMGCAS_tracing/182.img/raw/182.img.nii.gz',
        run_17_LV_segments = True
        ):

        combined_paths_folder = Path(folder_path) / "path_tracing" / "combined_paths"

        tree_paths = [p for p in combined_paths_folder.glob("*_traced_path_*.vtk")
                if not p.name.endswith("spline.vtk")]

        if run_17_LV_segments:


            wrap_lv_segments(
                    segmentation_path=segmentation_path,
                    image_path=image_path,
                    path=folder_path
                )

            extractor = TracerPointExtractor(tracer_folder_path=folder_path)
            extractor.return_tracer_points_as_json(atlas_json_path=folder_path / "misc" / "atlas.json",
                                                output_path=folder_path  / "tracer_points.json")

            save_distances_to_json(
                output_path / "tracer_points.json",
                output_path / "tracer_points_with_distances.json"
            )
            json_to_vtk_points(
            folder_path / "tracer_points.json",
            output_folder=folder_path / "vtk_output",
            prefix="tracer_points"
            )


        with open(output_path / "tracer_points_with_distances.json", 'r') as f:
            data = json.load(f)

        start_point = np.array(data["all_branch_points"][min(range(len(data["all_branch_points"])), key=lambda i: data["all_branch_points"][i][-1])][:3])
        start_point_idx = min(range(len(data["all_branch_points"])), key=lambda i: data["all_branch_points"][i][-1])

        point_set = [point[:3] for point in data["all_branch_points"]]
        filtered_point_set = [point for i, point in enumerate(point_set) if i != start_point_idx]

        #branch_points_idx = distance_between_point_and_set(start_point, point_set)

        tree_angles = {}

        for idx, tree_path in enumerate(tree_paths):
            print(f"Processing tree #{idx}: {tree_path}")
            # existing code that follows in your placeholder will run inside this loop
            match = re.search(r'traced_path_(\d+)_combined_path\.vtk', str(tree_path))
            if match:
                idx = int(match.group(1))
            else:
                idx = str(tree_path)
            v, v_bool = tree_run_through_points(
                tree_path=tree_path,
                json_path=output_path / "tracer_points.json"
            )



            matched_indices = np.where(v_bool == True)
            matched_point_set = [point_set[i] for i in matched_indices[0]]

            print(matched_point_set)


            branch_points_closest_to_root = distance_between_point_and_set(
                start_point,
                np.array(matched_point_set)
            )


            if branch_points_closest_to_root is None:
                tree_angles[idx] = None
                continue


            #branch_point_runthrough = np.argsort(v)[2:][0]




            # matched_value = np.intersect1d([branch_point_runthrough], branch_points_idx)
            tree = vtk_to_numpy(tree_path)
            point_clostest_to_start = distance_between_point_and_set(start_point, tree)[0]
            branch_match_point = tree[point_clostest_to_start+5]
            #branch_match_point = point_set[matched_indices[0][branch_points_closest_to_root]]

            apex = data["start_point"][0][:3]
            #np.array(data["atlas_points"][max(range(len(data["atlas_points"])), key=lambda i: data["atlas_points"][i][-1])][:3])
            vec_apex = vec_from_point_to_point(start_point, apex)
            vec_branch = vec_from_point_to_point(start_point, branch_match_point)

            # a = np.dot(vec_branch, vec_apex)
            # cos_theta = a / (np.linalg.norm(vec_branch) * np.linalg.norm(vec_apex))
            # angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            # angle_deg = np.degrees(angle_rad)
            cross = np.cross(vec_apex, vec_branch)


            tree_angles[idx] = cross[2]

            with open(folder_path / "tree_angles.json", 'w') as json_file:
                json.dump(tree_angles, json_file, indent=4)

        new_dict = label_tree_angles(folder_path)

        return new_dict

def label_tree_angles(output_path):
    tree_angles = output_path / "tree_angles.json"

    if tree_angles.exists():
        with tree_angles.open('r') as f:
            data = json.load(f)
    else:
        data = {}
    new_dict = {}
    new_dict["tree_angles"] = data
    max_min_values = max(data.values()), min(data.values())
    tree_labels = {
            key: "LAD" if angle == max_min_values[0] else "LCX" if angle == max_min_values[1] else "intermediate_value"
            for key, angle in data.items()
        }

    # tree_labels = {
    #         key: "LCX" if angle <= 0 else "LAD"
    #         for key, angle in data.items()
    #     }
    



    new_dict["tree_labels"] = tree_labels
    with tree_angles.open('w') as f:
        json.dump(new_dict, f, indent=4)

    return new_dict

def branch_point_detector(folder_path, threshold=0.2):
    combined_tree_files = list(folder_path.glob("path_tracing/combined_paths/*_combined_tree_spline.vtk"))
    spline_trees = list(folder_path.glob("path_tracing/combined_paths/*_combined_path_spline.vtk"))



    tree_paths = []

    combined_tree_files = vtk_to_numpy(combined_tree_files[0])
    tree_paths.append(combined_tree_files)

    for spline_tree in spline_trees:
        spline_tree_numpy = vtk_to_numpy(spline_tree)
        tree_paths.append(spline_tree_numpy)

    traced_paths = tree_paths[1:]
    # remove first 10 points from each traced path
    prune_n = 10
    traced_paths = [path[prune_n:] if len(path) > prune_n else path[:0] for path in traced_paths]


    if not traced_paths:
        return []


    num_trees = len(traced_paths)
    # Find the length of the longest path to define the number of comparisons
    max_len = max(len(path) for path in traced_paths)

    comparison_matrix = []

    # Iterate through each point index up to the max length
    for i in range(max_len):
        point_index_comparisons = []

        # Iterate through each "source" tree (T_j)
        for j in range(num_trees):
            # Check if the source tree has a point at this index
            if i >= len(traced_paths[j]):
                # If not, there's nothing to compare from, so add an empty tuple
                point_index_comparisons.append(tuple())
                continue

            source_point = traced_paths[j][i]
            comparison_results = []

            # Compare the source point with the point at the same index in all "target" trees (T_k)
            for k in range(num_trees):
                # Check if the target tree has a point at this index
                if i >= len(traced_paths[k]):
                    # If not, the comparison is not possible
                    comparison_results.append(False)
                    continue

                target_point = traced_paths[k][i]

                # Calculate Euclidean distance and check against the threshold
                distance = np.linalg.norm(source_point - target_point)
                is_within_threshold = distance < threshold
                comparison_results.append(is_within_threshold)

            point_index_comparisons.append(tuple(comparison_results))

        comparison_matrix.append(point_index_comparisons)

    return comparison_matrix

def find_min_distance_vectorized(A, B):
    """
    Vectorized approach using NumPy broadcasting.
    Faster than nested loops for moderate-sized arrays.
    """
    A = np.array(A)
    B = np.array(B)
    
    if A.size == 0 or B.size == 0:
        return float('inf')
    
    # Reshape A to (n, 1, 3) and B to (1, m, 3) for broadcasting
    # This creates a (n, m, 3) array of all pairwise differences
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    
    # Calculate squared distances: (n, m) array
    distances_sq = np.sum(diff**2, axis=2)
    
    # Find minimum distance
    return np.sqrt(np.min(distances_sq))

def branch_point_detector_backward_tracking(folder_path, distance_threshold=2.0, prune_end_n=5, tolerance=1.0, cluster_eps_multiplier=2.0, min_cluster_size=5):
    """
    Detects branch points by tracking backwards from endpoints and finding where paths converge.
    Redundancy handling: if path i meets path j at index idx_j (on path j),
    then when later analyzing path j, only analyze up to idx_j (inclusive) and stop.
    Also removes paths that are almost identical to others.

    Args:
        folder_path: Path to the folder containing traced paths
        distance_threshold: Maximum distance to consider paths as converging (in mm)
        prune_end_n: Number of endpoint points to skip (to avoid endpoint noise)
        tolerance: Tolerance for determining if two paths are almost identical

    Returns:
        np.ndarray: Detected branch point coordinates
    """
    def are_arrays_almost_identical(arr1, arr2, tolerance=1e-5):
        """Check if two arrays are almost identical within a given tolerance."""
        if arr1.shape != arr2.shape:
            return False
        return np.all(np.abs(arr1 - arr2) <= tolerance)

    spline_trees = list(folder_path.glob("path_tracing/combined_paths/*_combined_path_spline.vtk"))

    if len(spline_trees) < 2:
        print(f"Only {len(spline_trees)} paths found, need at least 2")
        return []

    # Load all paths (reversed so index 0 is endpoint after pruning)
    all_paths = []
    for spline_tree in spline_trees:
        points = vtk_to_numpy(spline_tree)
        if len(points) > prune_end_n:
            all_paths.append(points[::-1][prune_end_n:])
        else:
            all_paths.append(points[::-1][:0])  # empty

    # Early exit if nothing usable
    usable = [len(p) > 0 for p in all_paths]
    if sum(usable) < 2:
        print("Insufficient usable paths after pruning")
        return np.array([])

    print(f"Loaded {len(all_paths)} paths")

    # Remove redundant paths that are almost identical
    filtered_paths = []
    for i, path_i in enumerate(all_paths):
        is_redundant = False
        for j, path_j in enumerate(filtered_paths):
            if are_arrays_almost_identical(path_i[0], path_j[0], tolerance=tolerance):
                print(f"Path {i} is almost identical to another path; removing it")
                is_redundant = True
                break
        if not is_redundant:
            filtered_paths.append(path_i)

    all_paths = filtered_paths
    print(f"Filtered paths: {len(all_paths)} remaining after removing redundant paths")

    # Redundancy control: for each path, the max index (inclusive) to analyze.
    # None means analyze full length; otherwise stop at that index (inclusive).
    path_stop_indices = [None] * len(all_paths)

    convergence_data = []

    # For each "source" path
    for i, path_i in enumerate(all_paths):
        if len(path_i) == 0:
            continue

        # Determine allowed analysis limit for this path due to prior merges
        allowed_steps = len(path_i) - 1 if path_stop_indices[i] is None else min(path_stop_indices[i], len(path_i) - 1)
        if path_stop_indices[i] is not None:
            print(f"Limiting analysis for path {i} to step {allowed_steps} due to redundancy")

        print(f"\n=== Analyzing path {i} (length: {len(path_i)}, limit: {allowed_steps}) ===")

        # Track first encounters for this source path
        first_encounters = {}  # Change to store lists: {j: [encounter1, encounter2, ...]}

        # Iterate from endpoint towards start, but stop at redundancy limit
        for step_idx, point_i in enumerate(path_i):
            if step_idx > allowed_steps:
                break

            # Compare against all other paths
            for j, path_j in enumerate(all_paths):
                if i == j or len(path_j) == 0:
                    continue

                # Skip if we already have 5 or more encounters with path j
                if j in first_encounters and len(first_encounters[j]) >= 10:
                    continue

                # Compute distance to all points on path_j
                distances = np.linalg.norm(path_j - point_i, axis=1)
                min_distance = float(np.min(distances))
                if min_distance < distance_threshold:
                    closest_idx = int(np.argmin(distances))

                    # Record encounter
                    convergence_point = (point_i + path_j[closest_idx]) / 2.0
                    encounter = {
                        'step': int(step_idx),
                        'distance': float(min_distance),
                        'point': convergence_point,
                        'path_i_point': point_i.copy(),
                        'path_j_point': path_j[closest_idx].copy(),
                        'path_j_idx': int(closest_idx)
                    }
                    
                    # Add to list of encounters for this path pair
                    if j not in first_encounters:
                        first_encounters[j] = []
                    first_encounters[j].append(encounter)

                    print(f"  Path {i} meets path {j} at step {step_idx} (i) and {closest_idx} (j), distance: {min_distance:.2f}mm (encounter #{len(first_encounters[j])})")

                    # Redundancy rule: when later analyzing path j, stop at this index (inclusive)
                    if path_stop_indices[j] is None:
                        path_stop_indices[j] = 99999
                        print(f"  -> Redundancy set: path {j} stop at index {closest_idx}")
                    else:
                        if closest_idx < path_stop_indices[j]:
                            print(f"  -> Redundancy updated: path {j} stop {path_stop_indices[j]} -> {closest_idx}")
                            path_stop_indices[j] = 99999

        # Store convergence data for this path (flatten the encounter lists)
        if first_encounters:
            convergence_data.append({
                'path_id': i,
                'encounters': first_encounters  # Now contains lists of encounters
            })

    # Extract branch points from convergence data
    branch_points = []
    branch_metadata = []
    all_convergence_points = []
    for conv_data in convergence_data:
        for other_path_id, encounter_list in conv_data['encounters'].items():
            for encounter in encounter_list:
                all_convergence_points.append({
                    'point': encounter['point'],
                    'paths': (conv_data['path_id'], other_path_id),
                    'distance': encounter['distance']
            })

    if not all_convergence_points:
        print("No convergence points found")
        return np.array([])

    convergence_coords = np.array([cp['point'] for cp in all_convergence_points])

    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=distance_threshold * cluster_eps_multiplier, min_samples=min_cluster_size).fit(convergence_coords)

    for label in set(clustering.labels_):
        if label == -1:
            continue

        cluster_mask = clustering.labels_ == label
        cluster_points = convergence_coords[cluster_mask]
        cluster_data = [all_convergence_points[i] for i in np.where(cluster_mask)[0]]

        branch_point = np.mean(cluster_points, axis=0)

        involved_paths = set()
        for data in cluster_data:
            involved_paths.update(data['paths'])

        branch_points.append(branch_point)
        branch_metadata.append({
            'position': branch_point.tolist(),
            'num_paths': len(involved_paths),
            'path_ids': list(involved_paths),
            'cluster_size': len(cluster_points)
        })

        print(f"\nBranch point detected:")
        print(f"  Position: {branch_point}")
        print(f"  Paths involved: {list(involved_paths)}")
        print(f"  Number of convergences: {len(cluster_points)}")

    if branch_points:
        branch_points = np.array(branch_points)
        output_file = folder_path / "detected_branch_points_backward.vtk"
        numpy_to_vtk(branch_points, output_file)
        print(f"\nSaved {len(branch_points)} branch points to {output_file}")

        metadata_file = folder_path / "branch_points_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(branch_metadata, f, indent=4)
        print(f"Saved metadata to {metadata_file}")

        # Optionally persist stop indices info for debugging
        stop_info = {str(idx): (None if v is None else int(v)) for idx, v in enumerate(path_stop_indices)}
        with open(folder_path / "redundancy_stop_indices.json", 'w') as f:
            json.dump(stop_info, f, indent=4)
        print(f"Saved redundancy stop indices to {folder_path / 'redundancy_stop_indices.json'}")

        return branch_points.tolist()
    else:
        print("No branch points detected")
        return np.array([])

def validate_branch_points_divergence(folder_path, detected_branch_points, 
                                      distance_threshold=3.0, 
                                      trajectory_length=15,
                                      min_angle_threshold=30.0,
                                      min_diverging_paths=2):
    """
    Validates branch points by checking if paths actually DIVERGE after the branch point.
    Goes forward from start to end (opposite of detection).
    
    Args:
        folder_path: Path to folder containing traced paths
        detected_branch_points: np.ndarray of detected branch point coordinates
        distance_threshold: Max distance for a path to pass through branch point (mm)
        trajectory_length: Number of points to use for calculating direction after branch
        min_angle_threshold: Minimum angle between paths to consider them diverging (degrees)
        min_diverging_paths: Minimum number of paths that must diverge
        
    Returns:
        tuple: (valid_branch_points, validation_metadata)
    """
    import numpy as np
    from pathlib import Path
    
    spline_trees = list(folder_path.glob("path_tracing/combined_paths/*_combined_path_spline.vtk"))
    
    if len(spline_trees) < 2 or len(detected_branch_points) == 0:
        print("Insufficient data for validation")
        return np.array([]), []
    
    # Load all paths (NOT REVERSED - start to end)
    all_paths = []
    for spline_tree in spline_trees:
        points = vtk_to_numpy(spline_tree)
        all_paths.append(points)
    
    print(f"Validating {len(detected_branch_points)} branch points for divergence")
    
    valid_branch_points = []
    validation_metadata = []
    
    def calculate_trajectory_vector(path, start_idx, length):
        """Calculate direction vector from start_idx over 'length' points."""
        end_idx = min(start_idx + length, len(path))
        if end_idx <= start_idx + 1:
            return None
        # Direction = normalized vector from start to end of trajectory
        direction = path[end_idx - 1] - path[start_idx]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        return direction / norm
    
    def angle_between_vectors(v1, v2):
        """Calculate angle in degrees between two vectors."""
        if v1 is None or v2 is None:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    
    # Check each detected branch point
    for bp_idx, branch_point in enumerate(detected_branch_points):
        print(f"\n=== Validating branch point {bp_idx}: {branch_point} ===")
        
        path_trajectories = []
        
        # Find closest point on each path to this branch point
        for path_idx, path in enumerate(all_paths):
            distances = np.linalg.norm(path - branch_point, axis=1)
            min_distance = float(np.min(distances))
            closest_idx = int(np.argmin(distances))
            
            # Only consider paths that pass close to the branch point
            if min_distance < distance_threshold:
                # Calculate trajectory AFTER this point (going forward)
                trajectory_vec = calculate_trajectory_vector(path, closest_idx, trajectory_length)
                
                if trajectory_vec is not None:
                    path_trajectories.append({
                        'path_id': int(path_idx),
                        'closest_idx': int(closest_idx),
                        'min_distance': float(min_distance),
                        'trajectory': trajectory_vec.tolist(),  # Convert to list
                        'closest_point': [float(x) for x in path[closest_idx]]
                    })
                    print(f"  Path {path_idx}: distance={min_distance:.2f}mm, idx={closest_idx}")
        
        # Need at least 2 paths to check divergence
        if len(path_trajectories) < 2:
            validation_metadata.append({
                'position': [float(x) for x in branch_point],
                'is_valid': False,
                'reason': f'Only {len(path_trajectories)} path(s) pass through'
            })
            print(f"  ✗ INVALID: Only {len(path_trajectories)} path(s)")
            continue
        
        # Calculate pairwise angles between all trajectory vectors
        angles = []
        angle_pairs = []
        for i in range(len(path_trajectories)):
            for j in range(i + 1, len(path_trajectories)):
                angle = angle_between_vectors(
                    path_trajectories[i]['trajectory'],
                    path_trajectories[j]['trajectory']
                )
                angles.append(angle)
                angle_pairs.append({
                    'path_i': int(path_trajectories[i]['path_id']),
                    'path_j': int(path_trajectories[j]['path_id']),
                    'angle': float(angle)
                })
                print(f"  Angle between path {path_trajectories[i]['path_id']} and {path_trajectories[j]['path_id']}: {angle:.1f}°")
        
        # Check if enough paths diverge significantly
        significant_divergences = [a for a in angles if a >= min_angle_threshold]
        is_valid = len(significant_divergences) >= min_diverging_paths - 1  # n paths = n-1 pairs minimum
        
        if is_valid:
            valid_branch_points.append(branch_point)
            validation_metadata.append({
                'position': [float(x) for x in branch_point],
                'num_paths': int(len(path_trajectories)),
                'path_ids': [int(t['path_id']) for t in path_trajectories],
                'max_divergence_angle': float(max(angles)) if angles else 0.0,
                'mean_divergence_angle': float(np.mean(angles)) if angles else 0.0,
                'num_significant_divergences': int(len(significant_divergences)),
                'angle_pairs': angle_pairs,
                'paths_details': path_trajectories,
                'is_valid': True
            })
            print(f"  ✓ VALID: {len(significant_divergences)} significant divergences (max angle: {max(angles):.1f}°)")
        else:
            validation_metadata.append({
                'position': [float(x) for x in branch_point],
                'num_paths': int(len(path_trajectories)),
                'path_ids': [int(t['path_id']) for t in path_trajectories],
                'max_divergence_angle': float(max(angles)) if angles else 0.0,
                'num_significant_divergences': int(len(significant_divergences)),
                'is_valid': False,
                'reason': f'Only {len(significant_divergences)} significant divergences (need {min_diverging_paths - 1})'
            })
            print(f"  ✗ INVALID: Only {len(significant_divergences)} significant divergences")
    
    valid_branch_points = np.array(valid_branch_points) if valid_branch_points else np.array([])
    
    # Save results
    if len(valid_branch_points) > 0:
        output_file = folder_path / "validated_branch_points_divergence.vtk"
        numpy_to_vtk(valid_branch_points, output_file)
        print(f"\n✓ Saved {len(valid_branch_points)} validated branch points to {output_file}")
    
    metadata_file = folder_path / "validation_divergence_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(validation_metadata, f, indent=4)
    print(f"Saved validation metadata to {metadata_file}")
    
    print(f"\n=== DIVERGENCE VALIDATION SUMMARY ===")
    print(f"Total detected: {len(detected_branch_points)}")
    print(f"Valid branch points: {len(valid_branch_points)}")
    print(f"Invalid/filtered: {len(detected_branch_points) - len(valid_branch_points)}")
    
    return valid_branch_points.tolist(), validation_metadata

def visualize_vectors_angle(origin, vec1, vec2, output_path='vector_visualization.mrk.json'):
    """
    Creates a Slicer-compatible JSON file to visualize two vectors and the angle between them.
    
    Args:
        origin: np.ndarray - Starting point [x, y, z]
        vec1: np.ndarray - First vector [x, y, z]
        vec2: np.ndarray - Second vector [x, y, z]
        output_path: str - Output path for the Slicer markup JSON
    """
    import numpy as np
    import json
    
    origin = np.array(origin)
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate end points for the vectors
    end1 = origin + vec1
    end2 = origin + vec2
    
    # Calculate the angle
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # Create Slicer markup JSON
    markup = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [
            {
                "type": "Line",
                "coordinateSystem": "LPS",
                "locked": False,
                "label": f"Vector 1 (length: {np.linalg.norm(vec1):.2f})",
                "controlPoints": [
                    {"position": origin.tolist()},
                    {"position": end1.tolist()}
                ]
            },
            {
                "type": "Line",
                "coordinateSystem": "LPS",
                "locked": False,
                "label": f"Vector 2 (length: {np.linalg.norm(vec2):.2f})",
                "controlPoints": [
                    {"position": origin.tolist()},
                    {"position": end2.tolist()}
                ]
            },
            {
                "type": "Fiducial",
                "coordinateSystem": "LPS",
                "locked": False,
                "label": f"Origin",
                "controlPoints": [
                    {"position": origin.tolist()}
                ]
            },
            {
                "type": "Angle",
                "coordinateSystem": "LPS",
                "locked": False,
                "label": f"Angle: {angle_deg:.2f}°",
                "controlPoints": [
                    {"position": end1.tolist()},
                    {"position": origin.tolist()},
                    {"position": end2.tolist()}
                ]
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(markup, f, indent=2)
    
    print(f"\n=== Vector Angle Visualization ===")
    print(f"Origin: {origin}")
    print(f"Vector 1: {vec1} (magnitude: {np.linalg.norm(vec1):.2f})")
    print(f"Vector 2: {vec2} (magnitude: {np.linalg.norm(vec2):.2f})")
    print(f"Angle: {angle_deg:.2f}° ({angle_rad:.4f} radians)")
    print(f"Cosine: {cos_theta:.4f}")
    print(f"\nVisualization saved to: {output_path}")
    print("Load this file in 3D Slicer to see the vectors and angle")
    
    return angle_deg, output_path

if __name__ == "__main__":

    #id_list = ["130", "182", "281", "410", "519", "576", "654", "724", "802", "879"]
    #id_list = ["39", "119", "123", "124"]
    #com_points = load_atlas_json("assets/data/0010/processed/misc/atlas.json")
    compute_LV17 = True

    id_list = ["124"]
    for id in tqdm(id_list):
        # series_id = "0035"
        # CT_scan = f"CFA-PILOT_{id}_SERIES{series_id}"
        working_dir = Path.cwd()

        # # Construct the paths dynamically using pathlib
        # output_path = working_dir / f'assets/data/CoronaryTracing/{CT_scan}'
        # segmentation_path = working_dir / output_path / f'bartholinator/{CT_scan}_pred.nii.gz'
        # image_path = working_dir / output_path / f'raw/{CT_scan}.nii.gz'

        # output_path = working_dir / f'assets/data/IMGCAS_tracing/{id}.img'
        # segmentation_path = working_dir / output_path / f'bartholinator/{id}.img_pred.nii.gz'
        # image_path = working_dir / output_path / f'raw/{id}.img.nii.gz'
        # tree_path=output_path / "path_tracing" / "combined_paths" / f"{id}.img_traced_path_5_combined_path.vtk"

        output_path = working_dir / f'assets/imagecas/nii_images_sample_tracing/{id}.img'
        segmentation_path = working_dir / output_path / f'bartholinator/{id}.img_pred.nii.gz'
        image_path = working_dir / f'assets/imagecas/nii_images_sample/{id}.img.nii.gz'

        # output_path = working_dir / f'assets/data/0002'
        # segmentation_path = working_dir / output_path / f'bartholinator/CFA-PILOT_0002_SERIES0047_pred.nii.gz'
        # image_path = working_dir / output_path / f'raw/CFA-PILOT_0002_SERIES0047.nii.gz'


        # extractor = TracerPointExtractor(tracer_folder_path=output_path)
        # extractor.return_tracer_points_as_json(atlas_json_path=output_path / "misc" / "atlas.json",
        #                                     output_path=output_path  / "tracer_points.json")


        classify_trees_pipeline(
            folder_path=output_path,
            segmentation_path=segmentation_path,
            image_path=image_path,
            run_17_LV_segments=compute_LV17
            )

    #     detected_bp = branch_point_detector_backward_tracking(output_path, distance_threshold=0.35, prune_end_n=0, cluster_eps_multiplier=3.0, min_cluster_size=5)

    #     valid_bp, metadata = validate_branch_points_divergence(
    #         output_path,
    #         detected_bp,
    #         distance_threshold=0.5,
    #         trajectory_length=30,
    #         min_angle_threshold=15.0,
    #         min_diverging_paths=2
    #     )

    # print(f"\nFinal result: {len(valid_bp)} validated branch points")

    # if compute_LV17:

    #     wrap_lv_segments(
    #             segmentation_path=segmentation_path,
    #             image_path=image_path,
    #             path=output_path
    #         )

    # extractor = TracerPointExtractor(tracer_folder_path=output_path)
    # extractor.return_tracer_points_as_json(atlas_json_path=output_path / "misc" / "atlas.json",
    #                                        output_path=output_path  / "tracer_points.json")


    # #vtk_to_numpy("F:/samT7/sp/assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036/path_tracing/combined_paths/CFA-PILOT_0010_SERIES0036_traced_path_1_combined_path.vtk")


    # # print("Start Point:", points["start_point"])
    # # print("End Points:", points["end_points"])
    # # print("All Branch Points:", points["all_branch_points"])


    # # json_to_vtk_points(
    # #     "output.json",
    # #     output_folder="vtk_output",
    # #     prefix="tracer_points"
    # # )

    # save_distances_to_json(
    #     output_path / "tracer_points.json",
    #     output_path / "tracer_points_with_distances.json"
    # )
    # json_to_vtk_points(
    # output_path / "tracer_points.json",
    # output_folder=output_path / "vtk_output",
    # prefix="tracer_points"
    # )

    # # vectors_to_root(type_points="atlas_points", load_path=output_path / "tracer_points.json")

    # # visualize_slicer_vecs(
    # #                     load_path=output_path / "tracer_points_with_distances.json",
    # #                     output_path=output_path / "branch_to_root_lines.mrk.json",
    # #                     type_points="atlas_points"
    # # )

    # with open(output_path / "tracer_points_with_distances.json", 'r') as f:
    #     data = json.load(f)

    # start_point = np.array(data["all_branch_points"][min(range(len(data["all_branch_points"])), key=lambda i: data["all_branch_points"][i][-1])][:3])
    # start_point_idx = min(range(len(data["all_branch_points"])), key=lambda i: data["all_branch_points"][i][-1])

    # point_set = [point[:3] for point in data["all_branch_points"]]
    # filtered_point_set = [point for i, point in enumerate(point_set) if i != start_point_idx]

    # #branch_points_idx = distance_between_point_and_set(start_point, point_set)

    # v, v_bool = tree_run_through_points(
    #     tree_path=tree_path,
    #     json_path=output_path / "tracer_points.json"
    # )
    # matched_indices = np.where(v_bool == True)
    # matched_point_set = [point_set[i] for i in matched_indices[0]]

    # print(matched_point_set)


    # branch_points_closest_to_root = distance_between_point_and_set(
    #     start_point,
    #     np.array(matched_point_set)
    # )



    # print(1)

    # #branch_point_runthrough = np.argsort(v)[2:][0]




    # # matched_value = np.intersect1d([branch_point_runthrough], branch_points_idx)

    # branch_match_point = point_set[matched_indices[0][branch_points_closest_to_root]]

    # apex = np.array(data["atlas_points"][max(range(len(data["atlas_points"])), key=lambda i: data["atlas_points"][i][-1])][:3])
    # vec_apex = vec_from_point_to_point(start_point, apex)
    # vec_branch = vec_from_point_to_point(start_point, branch_match_point)

    # a = np.dot(vec_branch, vec_apex)
    # cos_theta = a / (np.linalg.norm(vec_branch) * np.linalg.norm(vec_apex))
    # angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    # angle_deg = np.degrees(angle_rad)
    # print("Angle between branch and apex vectors:", angle_deg)


    # # branch_point_1 = point_set[branch_points_idx[0]][:3]
    # # branch_point_2 = point_set[branch_points_idx[1]][:3]
    # # apex = np.array(data["atlas_points"][max(range(len(data["atlas_points"])), key=lambda i: data["atlas_points"][i][-1])][:3])

    # # vec1 = vec_from_point_to_point(start_point, branch_point_1)
    # # vec2 = vec_from_point_to_point(start_point, branch_point_2)
    # # vec3 = vec_from_point_to_point(start_point, apex)

    # # a1 = np.dot(vec1, vec3)
    # # a2 = np.dot(vec2, vec3)

    # # cos_theta1 = a1 / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    # # cos_theta2 = a2 / (np.linalg.norm(vec2) * np.linalg.norm(vec3))

    # # # Calculate the actual angles in radians
    # # angle1_rad = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
    # # angle2_rad = np.arccos(np.clip(cos_theta2, -1.0, 1.0))

    # # # Convert to degrees for easier interpretation
    # # angle1_deg = np.degrees(angle1_rad)
    # # angle2_deg = np.degrees(angle2_rad)

    # # print("Cosine of angle between vec1 and vec3:", cos_theta1)
    # # print("Angle between vec1 and vec3:", angle1_deg, "degrees")
    # # print("Cosine of angle between vec2 and vec3:", cos_theta2)
    # # print("Angle between vec2 and vec3:", angle2_deg, "degrees")





    # convert_txt_to_vtk(
    #     "test123.txt",
    #     "test123.vtk"
    # )

    # #get_last_point_from_vtk("F:/samT7/sp/assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036/path_tracing/tracing_with_momentum_individual_paths/CFA-PILOT_0010_SERIES0036_traced_path_2.vtk")
