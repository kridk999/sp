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


def get_last_point_from_vtk(vtk_file_path):
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

    # Return the last point
    return points[-1] if points else None


class TracerPointExtractor:
    def __init__(self,
                 tracer_folder_path : str = "assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036"):
        
        self.tracer_folder_path = tracer_folder_path
        self.combined_paths_folder = 123



if __name__ == "__main__":

    convert_txt_to_vtk(
        "test321.txt",
        "test321.vtk"
    )   
    
    #get_last_point_from_vtk("F:/samT7/sp/assets/data/CoronaryTracing/CFA-PILOT_0010_SERIES0036/path_tracing/tracing_with_momentum_individual_paths/CFA-PILOT_0010_SERIES0036_traced_path_2.vtk")
