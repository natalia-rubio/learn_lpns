#!/usr/bin/env python3
"""
Script to visualize 3D solution results as a movie.
Shows the 3D mesh colored by pressure or velocity over time.
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


def visualize_3d_solution(vtu_path, output_path, field='pressure', 
                          resolution=(1920, 1080), fps=5):
    """
    Create a movie visualizing 3D solution from VTU file.
    
    Args:
        vtu_path: Path to 3D solution VTU file (combined_results.vtu)
        output_path: Path to save output MP4
        field: Field to visualize ('pressure' or 'velocity')
        resolution: (width, height) for output video
        fps: Frames per second
    """
    print("="*60)
    print("Visualizing 3D Solution")
    print("="*60)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg is required but not found.")
        return False
    
    # Read 3D solution VTU
    print(f"\nReading 3D solution from: {vtu_path}")
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    mesh = reader.GetOutput()
    print(f"  Mesh has {mesh.GetNumberOfPoints()} points")
    print(f"  Mesh has {mesh.GetNumberOfCells()} cells")
    
    # Extract point data arrays
    point_data = mesh.GetPointData()
    
    # Find all timestep arrays for the requested field
    timestep_arrays = []
    field_keywords = ['velocity'] if field == 'velocity' else ['pressure']
    
    print(f"\nSearching for {field} arrays...")
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        if array_name:
            print(f"  Found array: {array_name}")
            for keyword in field_keywords:
                if array_name.startswith(f'{keyword}_'):
                    timestep_arrays.append(array_name)
                    break
    
    if not timestep_arrays:
        print(f"Error: No {field} timestep arrays found in 3D solution")
        print(f"  Available arrays:")
        for i in range(point_data.GetNumberOfArrays()):
            print(f"    - {point_data.GetArrayName(i)}")
        return False
    
    # Sort timesteps
    def extract_timestep(name):
        try:
            # Handle formats like "pressure_00001" or "pressure_1"
            parts = name.split('_')
            if len(parts) > 1:
                return int(parts[-1])
            return 0
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
            # Handle vector arrays (velocity has 3 components)
            if array.GetNumberOfComponents() > 1:
                # Use magnitude for vector fields
                if array.GetNumberOfComponents() == 3:
                    values = np.sqrt(values[:, 0]**2 + values[:, 1]**2 + values[:, 2]**2)
                else:
                    values = np.linalg.norm(values, axis=1)
            all_values.extend(values.tolist())
    
    if not all_values:
        print(f"Error: No values found for {field}")
        return False
    
    color_range = [np.min(all_values), np.max(all_values)]
    print(f"  Color range: [{color_range[0]:.2f}, {color_range[1]:.2f}]")
    
    # Create mapper and actor for the mesh
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(mesh)
    mapper.SetScalarModeToUsePointData()
    mapper.SetScalarRange(color_range[0], color_range[1])
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    
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
    temp_dir = tempfile.mkdtemp(prefix='threed_viz_')
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
        
        mapper.SetLookupTable(lut)
        
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
            print(f"  Frame {idx+1}/{num_timesteps}: timestep {timestep} ({array_name})")
            
            # Set scalar data for this timestep
            array = point_data.GetArray(array_name)
            if array is None:
                print(f"    Warning: Array {array_name} not found, skipping frame")
                continue
            
            # Set active scalars
            point_data.SetActiveScalars(array_name)
            mapper.Update()
            
            # Remove previous text actor
            if text_actor is not None:
                renderer.RemoveActor2D(text_actor)
            
            # Add time text
            # Calculate normalized time (assuming timesteps are evenly spaced in [0, 1])
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
        description="Visualize 3D solution results as a movie"
    )
    parser.add_argument('--set-name', required=True, help='Set name (e.g., set_1)')
    parser.add_argument('--geo-name', required=True, help='Geometry name (e.g., tree_000)')
    parser.add_argument('--vtu-file', help='Path to 3D solution VTU file (default: auto-detect)')
    parser.add_argument('--field', choices=['pressure', 'velocity'], default='pressure',
                       help='Field to visualize (default: pressure)')
    parser.add_argument('--output', help='Output movie path (default: auto-generate)')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--output-dir', default='data/zeroD', help='Output directory (default: data/zeroD)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'), help='Video resolution (default: 1920 1080)')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second (default: 5)')
    
    args = parser.parse_args()
    
    # Auto-detect VTU file
    if args.vtu_file:
        vtu_path = args.vtu_file
    else:
        vtu_path = os.path.join(args.data_dir, 'threeD', args.set_name, args.geo_name, 'combined_results.vtu')
        if not os.path.exists(vtu_path):
            print(f"Error: 3D solution file not found: {vtu_path}")
            sys.exit(1)
    
    # Auto-generate output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(args.output_dir, args.set_name, args.geo_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.geo_name}_3d_{args.field}.mp4")
    
    print("="*60)
    print("3D Solution Visualization")
    print("="*60)
    print(f"  3D solution: {vtu_path}")
    print(f"  Field: {args.field}")
    print(f"  Output: {output_path}")
    print("="*60)
    
    success = visualize_3d_solution(
        vtu_path, output_path,
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

