from pathlib import Path
import glob
import json

import numpy as np

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
        for line in lines[2:]:
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

        self.all_bp = self._parse_xyz_coordinates(
                    self._get_all_bp_points()
        )


    def _get_amount_of_traced_paths(self):

    
        return len([p for p in self.combined_paths_folder.glob("CFA-PILOT_*_traced_path_*.vtk")
            if not p.name.endswith("spline.vtk")])
    
    def _get_traced_path_files(self):

    
        return [p for p in self.combined_paths_folder.glob("CFA-PILOT_*_traced_path_*.vtk")
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

    def _get_all_bp_points(self):
        bp_point_files = glob.glob(str(self.individual_paths_folder / "*all_bp.txt"))

        with open(bp_point_files[0], 'r') as vtk_file:
            return vtk_file.read().strip()  # Read and store the content

    def _parse_xyz_coordinates(self, coordinate_string):

        # Split the string into lines
        lines = coordinate_string.strip().split("\n")
        
        # Convert each line into a list of floats
        coordinates = [list(map(float, line.split())) for line in lines]
        
        return coordinates

    def extract_tracer_points(self):
        return {
            "start_point": self.start_point,
            "end_points": self.end_points,
            "all_branch_points": self.all_bp
        }

    def return_tracer_points_as_json(self):
        
        tracer_points = self.extract_tracer_points()

        output_file = "output.json"
        with open(output_file, "w") as json_file:
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


if __name__ == "__main__":

    # extractor = TracerPointExtractor()
    # points = extractor.extract_tracer_points()
    # extractor.return_tracer_points_as_json()
    # print("Start Point:", points["start_point"])
    # print("End Points:", points["end_points"])
    # print("All Branch Points:", points["all_branch_points"])


    # json_to_vtk_points(
    #     "output.json",
    #     output_folder="vtk_output",
    #     prefix="tracer_points"
    # )

    # save_distances_to_json(
    #     "output.json",
    #     "output_with_distances.json"
    # )

    convert_txt_to_vtk(
        "\\Users\\jacob\\OneDrive\\Uni\\9. Semester\\SpecialProjekt\\sp\\assets\\data\\0010\\processed\\misc\\atlas_points.txt",
        "test123.vtk"
    )   
    
    #get_last_point_from_vtk("F:/samT7/sp/assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036/path_tracing/tracing_with_momentum_individual_paths/CFA-PILOT_0010_SERIES0036_traced_path_2.vtk")
