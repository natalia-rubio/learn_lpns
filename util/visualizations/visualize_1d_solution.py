#!/usr/bin/env python3
"""
Script to visualize 1D centerline solution as a movie.
Shows the centerline colored by pressure or flow over time.
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess

# Set up for off-screen rendering
os.environ['VTK_USE_OFFSCREEN'] = '1'

try:
    import vtk
    vtk.vtkRenderWindow.GlobalWarningDisplayOff()
    from vtk.util.numpy_support import vtk_to_numpy as v2n
except ImportError:
    print("Error: VTK is required. Make sure VTK is installed and available.")
    sys.exit(1)

import numpy as np


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def create_vasculature_actor(mesh_path):
    """
    Create a VTK actor for the 3D vasculature mesh (gray, half opacity).
    
    Returns:
        actor: VTK actor
    """
    # Read mesh
    if mesh_path.endswith('.vtu'):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif mesh_path.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
    
    reader.SetFileName(mesh_path)
    reader.Update()
    
    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    mapper.ScalarVisibilityOff()  # Use solid color
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Set gray color and half opacity
    actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Gray
    actor.GetProperty().SetOpacity(0.5)
    
    return actor


def combine_frames_to_movie(image_files, output_path, fps, resolution):
    """
    Combine image frames into a movie using ffmpeg.
    
    Args:
        image_files: List of image file paths
        output_path: Output movie path
        fps: Frames per second
        resolution: (width, height) tuple
        
    Returns:
        True if successful, False otherwise
    """
    if not image_files:
        print("Error: No frames to combine.")
        return False
    
    print(f"\nCombining {len(image_files)} frames into movie...")
    print(f"  Output: {output_path}")
    
    # Try different encoders
    encoders = [
        ('libx264', ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']),
        ('h264_videotoolbox', ['-c:v', 'h264_videotoolbox', '-b:v', '5M', '-allow_sw', '1']),
        ('libopenh264', ['-c:v', 'libopenh264', '-pix_fmt', 'yuv420p']),
        ('libvpx', ['-c:v', 'libvpx', '-b:v', '2M', '-pix_fmt', 'yuv420p']),
    ]
    
    # Use pattern matching for input files
    temp_dir = os.path.dirname(image_files[0])
    frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
    
    success = False
    for encoder_name, encoder_args in encoders:
        print(f"  Trying encoder: {encoder_name}...")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-vf', f'scale={resolution[0]}:{resolution[1]}',
        ] + encoder_args + [output_path]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print(f"  Movie created successfully using {encoder_name}!")
            success = True
            break
        except subprocess.CalledProcessError as e:
            if 'encoder' in e.stderr.lower() or 'Unknown encoder' in e.stderr:
                print(f"    Encoder {encoder_name} not available, trying next...")
                continue
            else:
                print(f"    Error with {encoder_name}: {e.stderr[:200]}")
                continue
    
    if not success:
        print("Error: All encoders failed.")
        return False
    
    return True


def visualize_1d_solution(oned_soln_path, centerline_path, mesh_path, output_path,
                           field='pressure', resolution=(1920, 1080), fps=5):
    """
    Create a movie visualizing 1D centerline solution.
    
    Args:
        oned_soln_path: Path to 1D solution VTP file (unsteady_soln.vtp)
        centerline_path: Path to centerline VTP file (for geometry)
        mesh_path: Path to 3D mesh VTU/VTP file (for background)
        output_path: Path to save output MP4
        field: Field to visualize ('pressure' or 'flow'/'velocity')
        resolution: (width, height) for output video
        fps: Frames per second
    """
    print("="*60)
    print("Visualizing 1D Centerline Solution")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        return False
    
    # Read 1D solution VTP
    print(f"\nReading 1D solution from: {oned_soln_path}")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(oned_soln_path)
    reader.Update()
    centerline_polydata = reader.GetOutput()
    print(f"  Centerline has {centerline_polydata.GetNumberOfPoints()} points")
    
    # Extract point data arrays
    point_data = centerline_polydata.GetPointData()
    
    # Find all timestep arrays for the requested field
    timestep_arrays = []
    field_keywords = ['velocity'] if field == 'flow' else ['pressure']
    
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        for keyword in field_keywords:
            if array_name and array_name.startswith(f'{keyword}_'):
                timestep_arrays.append(array_name)
                break
    
    if not timestep_arrays:
        print(f"Error: No {field} timestep arrays found in 1D solution")
        print(f"  Available arrays:")
        for i in range(point_data.GetNumberOfArrays()):
            print(f"    - {point_data.GetArrayName(i)}")
        return False
    
    # Sort timesteps
    def extract_timestep(name):
        try:
            return int(name.split('_')[-1])
        except:
            return 0
    
    timestep_arrays.sort(key=extract_timestep)
    total_timesteps = len(timestep_arrays)
    print(f"  Found {total_timesteps} timesteps for {field}")
    
    # Filter to only the second half of the time period
    if total_timesteps > 1:
        mid_point = total_timesteps // 2
        timestep_arrays = timestep_arrays[mid_point:]
        num_timesteps = len(timestep_arrays)
        print(f"  Using second half of time period: {num_timesteps} timesteps (from timestep {extract_timestep(timestep_arrays[0])} to {extract_timestep(timestep_arrays[-1])})")
    else:
        num_timesteps = total_timesteps
    
    # Extract all values for color range
    print(f"\nComputing color range for {field}...")
    all_values = []
    for array_name in timestep_arrays:
        array = point_data.GetArray(array_name)
        if array:
            values = v2n(array)
            all_values.extend(values.tolist())
    
    if not all_values:
        print(f"Error: No values found for {field}")
        return False
    
    color_range = [np.min(all_values), np.max(all_values)]
    print(f"  Color range: [{color_range[0]:.2f}, {color_range[1]:.2f}]")
    
    # Create vasculature actor for background
    print("\nLoading 3D vasculature mesh...")
    vasculature_actor = create_vasculature_actor(mesh_path)
    
    # Create centerline actor (will be updated each frame)
    centerline_mapper = vtk.vtkPolyDataMapper()
    centerline_mapper.SetInputData(centerline_polydata)
    centerline_actor = vtk.vtkActor()
    centerline_actor.SetMapper(centerline_mapper)
    centerline_actor.GetProperty().SetLineWidth(5.0)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(vasculature_actor)
    renderer.AddActor(centerline_actor)
    
    # Set up camera
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Zoom(0.9)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(resolution[0], resolution[1])
    render_window.SetOffScreenRendering(1)
    render_window.SetShowWindow(False)
    
    try:
        render_window.Initialize()
    except Exception as e:
        print(f"Error: Render window initialization failed: {e}")
        return False
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='oned_viz_')
    print(f"\nRendering {num_timesteps} frames...")
    print(f"  Temporary directory: {temp_dir}")
    
    try:
        # Create lookup table
        lut = vtk.vtkLookupTable()
        lut.SetTableRange(color_range[0], color_range[1])
        lut.Build()
        # Reverse colormap (red high, blue low)
        num_colors = lut.GetNumberOfTableValues()
        temp_colors = []
        for i in range(num_colors):
            r, g, b, a = lut.GetTableValue(i)
            temp_colors.append((r, g, b, a))
        for i in range(num_colors):
            r, g, b, a = temp_colors[num_colors - 1 - i]
            lut.SetTableValue(i, r, g, b, a)
        
        centerline_mapper.SetLookupTable(lut)
        centerline_mapper.SetScalarRange(color_range[0], color_range[1])
        centerline_mapper.SetScalarModeToUsePointData()
        
        # Create colorbar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetTitle(f"{field.capitalize()}")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetLookupTable(lut)
        scalar_bar.GetTitleTextProperty().SetFontSize(32)
        scalar_bar.GetLabelTextProperty().SetFontSize(28)
        renderer.AddActor2D(scalar_bar)
        
        # Track text actor
        text_actor = None
        
        image_files = []
        for idx, array_name in enumerate(timestep_arrays):
            timestep = extract_timestep(array_name)
            print(f"  Frame {idx+1}/{num_timesteps}: timestep {timestep}")
            
            # Set scalar data for this timestep
            array = point_data.GetArray(array_name)
            if array:
                centerline_polydata.GetPointData().SetScalars(array)
                centerline_polydata.GetPointData().SetActiveScalars(array_name)
                centerline_mapper.Update()
            
            # Remove previous text actor
            if text_actor is not None:
                renderer.RemoveActor2D(text_actor)
            
            # Add time text
            # Calculate normalized time (assuming timesteps are evenly spaced in [0, 1])
            # Use original timestep number to get correct normalized time
            normalized_time = timestep / (total_timesteps - 1) if total_timesteps > 1 else 0.0
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"Time: {normalized_time:.3f} s\nField: {field.capitalize()}")
            text_actor.SetPosition(10, resolution[1] - 60)
            text_prop = text_actor.GetTextProperty()
            text_prop.SetFontSize(20)
            text_prop.SetColor(0.0, 0.0, 0.0)
            renderer.AddActor2D(text_actor)
            
            # Render frame
            render_window.Render()
            
            # Save frame
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(render_window)
            window_to_image.Update()
            
            image_file = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(image_file)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()
            image_files.append(image_file)
        
        # Remove text actor
        if text_actor is not None:
            renderer.RemoveActor2D(text_actor)
        
        # Combine frames into movie
        success = combine_frames_to_movie(image_files, output_path, fps, resolution)
        
        if success:
            print(f"✓ Movie saved to: {output_path}")
            return True
        else:
            print(f"✗ Failed to create movie")
            return False
            
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 1D centerline solution as a movie"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_000)')
    parser.add_argument('--oned-soln', help='Path to 1D solution VTP file (default: auto-detect)')
    parser.add_argument('--centerline', help='Path to centerline VTP file (default: auto-detect)')
    parser.add_argument('--mesh', help='Path to 3D mesh VTU/VTP file (default: auto-detect)')
    parser.add_argument('--field', choices=['pressure', 'flow'], default='pressure',
                       help='Field to visualize (default: pressure)')
    parser.add_argument('--output', help='Output movie path (default: auto-generate)')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--output-dir', default='data/zeroD', help='Output directory (default: data/zeroD)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'), help='Video resolution (default: 1920 1080)')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second (default: 5)')
    
    args = parser.parse_args()
    
    # Auto-detect 1D solution file
    if args.oned_soln:
        oned_soln_path = args.oned_soln
    else:
        # Try different locations
        paths = [
            os.path.join(args.data_dir, 'oneD', args.set_name, args.geo_name, 'unsteady_soln.vtp'),
            os.path.join(args.data_dir, 'reduced_results', args.set_name, args.geo_name, 'unsteady_soln.vtp'),
        ]
        oned_soln_path = None
        for path in paths:
            if os.path.exists(path):
                oned_soln_path = path
                break
        
        if oned_soln_path is None:
            print(f"Error: 1D solution file not found. Tried:")
            for path in paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Auto-detect centerline
    if args.centerline:
        centerline_path = args.centerline
    else:
        centerline_path = os.path.join(args.data_dir, 'threeD', args.set_name, args.geo_name, 'centerlines_simVascular.vtp')
        if not os.path.exists(centerline_path):
            print(f"Error: Centerline file not found: {centerline_path}")
            sys.exit(1)
    
    # Auto-detect mesh
    if args.mesh:
        mesh_path = args.mesh
    else:
        mesh_paths = [
            os.path.join(args.data_dir, 'threeD', args.set_name, args.geo_name, 'mesh', 'fluid_msh_0', 'fluid_msh_0.vtu'),
            os.path.join(args.data_dir, 'threeD', args.set_name, args.geo_name, 'mesh-complete', 'mesh-complete.mesh.vtu'),
        ]
        mesh_path = None
        for path in mesh_paths:
            if os.path.exists(path):
                mesh_path = path
                break
        
        if mesh_path is None:
            print(f"Error: Mesh file not found. Tried:")
            for path in mesh_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.geo_name}_1d_{args.field}.mp4")
    
    print("="*60)
    print("1D Solution Visualization")
    print("="*60)
    print(f"  1D solution: {oned_soln_path}")
    print(f"  Centerline: {centerline_path}")
    print(f"  Mesh: {mesh_path}")
    print(f"  Field: {args.field}")
    print(f"  Output: {output_path}")
    print("="*60)
    
    success = visualize_1d_solution(
        oned_soln_path, centerline_path, mesh_path, output_path,
        args.field, tuple(args.resolution), args.fps
    )
    
    if success:
        print(f"\n✓ Successfully created movie: {output_path}")
        sys.exit(0)
    else:
        print(f"\n✗ Failed to create movie")
        sys.exit(1)


if __name__ == '__main__':
    main()
    
# python3 util/visualizations/visualize_1d_solution.py --set-name set_3 --geo-name tree_007 --field flow

