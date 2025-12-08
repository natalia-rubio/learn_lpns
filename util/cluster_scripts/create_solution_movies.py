#!/usr/bin/env python3
"""
Script to generate movies from 3D VTU and 1D centerline VTP solution files.
Supports pressure and velocity visualization for both 3D and 1D solutions.
"""

import os
import sys
import glob
import argparse
import tempfile
import shutil
from pathlib import Path

# Check numpy version first (pyvista requires numpy >= 1.21.0, but scipy 1.6.3 requires < 1.23.0)
try:
    import numpy as np
    numpy_version = np.__version__
    numpy_major, numpy_minor = map(int, numpy_version.split('.')[:2])
    if numpy_major < 1 or (numpy_major == 1 and numpy_minor < 21):
        print(f"Error: numpy version {numpy_version} is too old. pyvista requires numpy >= 1.21.0")
        print("Please upgrade numpy: pip install --upgrade --user 'numpy>=1.21.0,<1.23.0'")
        sys.exit(1)
    elif numpy_major >= 2:
        print(f"Warning: numpy version {numpy_version} may be incompatible with scipy 1.6.3")
        print("Consider using: pip install --upgrade --user 'numpy>=1.21.0,<1.23.0'")
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import pyvista as pv
    pv.set_plot_theme("document")
except ImportError as e:
    print("Error: pyvista is required. Install with: pip install pyvista")
    print(f"Import error: {e}")
    sys.exit(1)
except AttributeError as e:
    if "NDArray" in str(e):
        print("Error: Version mismatch between numpy and pyvista.")
        print("Please upgrade numpy: pip install --upgrade 'numpy>=1.21.0'")
        print(f"Current numpy version: {numpy_version}")
        print(f"Error details: {e}")
    else:
        print(f"Error importing pyvista: {e}")
    sys.exit(1)

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    print("Error: moviepy is required. Install with: pip install moviepy")
    sys.exit(1)

def pad_timestep(time_str, width=5):
    """Pad timestep string with zeros."""
    try:
        return str(int(time_str)).zfill(width)
    except (ValueError, TypeError):
        return time_str

def find_timestep_files(directory, pattern="result_*.vtu"):
    """
    Find all timestep files in directory.
    Returns sorted list of (timestep_str, filepath) tuples.
    """
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return []
    
    timesteps = []
    for f in files:
        basename = os.path.basename(f)
        # Extract timestep from result_XXX.vtu format
        if basename.startswith('result_') and basename.endswith('.vtu'):
            time_str = basename.replace('result_', '').replace('.vtu', '')
            if time_str.isdigit() or (time_str and all(c.isdigit() for c in time_str)):
                timesteps.append((time_str, f))
    
    timesteps.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    return timesteps

def create_3d_movie(vtu_files, output_path, scalar_field="pressure", 
                    slice_normal=[1, 0, 0], clim=None, fps=10, 
                    resolution=(1920, 1080), show_mesh=False):
    """
    Create movie from 3D VTU files.
    
    Args:
        vtu_files: List of (timestep, filepath) tuples
        output_path: Output movie file path
        scalar_field: Field to visualize ("pressure" or "velocity")
        slice_normal: Normal vector for slicing [x, y, z]
        clim: Color limits [min, max] or None for auto
        fps: Frames per second
        resolution: (width, height) for output
        show_mesh: If True, show full mesh; if False, show slice only
    """
    if not vtu_files:
        print("  No VTU files found")
        return False
    
    print(f"  Creating 3D movie from {len(vtu_files)} timesteps...")
    print(f"  Field: {scalar_field}, Output: {output_path}")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="movie_frames_")
    image_files = []
    
    try:
        # Read first file to get mesh structure and determine color limits
        first_file = vtu_files[0][1]
        mesh = pv.read(first_file)
        
        # Determine color limits if not provided
        if clim is None:
            all_values = []
            for time_str, vtu_file in vtu_files[:min(10, len(vtu_files))]:  # Sample first 10
                try:
                    temp_mesh = pv.read(vtu_file)
                    field_name = None
                    for name in temp_mesh.array_names:
                        if scalar_field.lower() in name.lower():
                            field_name = name
                            break
                    if field_name:
                        values = temp_mesh[field_name]
                        if values.ndim == 1:  # Scalar field
                            all_values.extend(values)
                        else:  # Vector field - use magnitude
                            all_values.extend(np.linalg.norm(values, axis=1))
                except:
                    continue
            if all_values:
                clim = [np.percentile(all_values, 5), np.percentile(all_values, 95)]
            else:
                clim = [0, 1]
        
        print(f"  Color limits: {clim}")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        
        # Process each timestep
        for idx, (time_str, vtu_file) in enumerate(vtu_files):
            try:
                mesh = pv.read(vtu_file)
                
                # Find the field array name
                field_name = None
                for name in mesh.array_names:
                    if scalar_field.lower() in name.lower():
                        field_name = name
                        break
                
                if field_name is None:
                    print(f"    Warning: Field '{scalar_field}' not found in {vtu_file}")
                    continue
                
                # Clear previous actors
                plotter.clear()
                
                # Create visualization
                if show_mesh:
                    # Show full mesh
                    plotter.add_mesh(mesh, scalars=field_name, clim=clim, 
                                   cmap='viridis', show_edges=False)
                else:
                    # Show slice
                    slice_mesh = mesh.slice(normal=slice_normal)
                    if slice_mesh.n_points > 0:
                        plotter.add_mesh(slice_mesh, scalars=field_name, clim=clim,
                                       cmap='viridis', show_edges=False)
                
                # Add title with timestep
                time_padded = pad_timestep(time_str)
                plotter.add_text(f"Timestep: {time_padded}\nField: {scalar_field}", 
                               font_size=20, position='upper_left')
                
                # Add colorbar
                plotter.add_scalar_bar(title=scalar_field.capitalize(), 
                                     vertical=True, title_font_size=16)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                plotter.screenshot(frame_path)
                image_files.append(frame_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"    Processed {idx + 1}/{len(vtu_files)} frames...")
                    
            except Exception as e:
                print(f"    Warning: Failed to process {vtu_file}: {e}")
                continue
        
        plotter.close()
        
        if not image_files:
            print("  Error: No frames generated")
            return False
        
        # Create movie from frames
        print(f"  Creating movie from {len(image_files)} frames...")
        clip = ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(output_path, fps=fps, codec='libx264', 
                           bitrate='8000k', audio=False)
        clip.close()
        
        print(f"  Movie saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error creating 3D movie: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_1d_movie(vtp_file, output_path, scalar_field="pressure",
                    fps=10, resolution=(1920, 1080), line_width=5):
    """
    Create movie from 1D centerline VTP file with multiple timesteps.
    
    Args:
        vtp_file: Path to centerline VTP file with timestep arrays
        output_path: Output movie file path
        scalar_field: Field to visualize ("pressure" or "velocity")
        fps: Frames per second
        resolution: (width, height) for output
        line_width: Width of centerline tube
    """
    if not os.path.exists(vtp_file):
        print(f"  Error: VTP file not found: {vtp_file}")
        return False
    
    print(f"  Creating 1D movie from: {vtp_file}")
    print(f"  Field: {scalar_field}, Output: {output_path}")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="movie_frames_")
    image_files = []
    
    try:
        # Read centerline
        centerline = pv.read(vtp_file)
        
        # Find all timestep arrays for the field
        field_arrays = []
        for name in centerline.array_names:
            if scalar_field.lower() in name.lower() and '_' in name:
                # Extract timestep from name like "pressure_00002"
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        time_str = parts[-1]
                        int(time_str)  # Verify it's numeric
                        field_arrays.append((time_str, name))
                    except ValueError:
                        continue
        
        if not field_arrays:
            print(f"  Error: No timestep arrays found for field '{scalar_field}'")
            return False
        
        field_arrays.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        print(f"  Found {len(field_arrays)} timesteps")
        
        # Determine color limits
        all_values = []
        for time_str, array_name in field_arrays:
            values = centerline[array_name]
            if values is not None:
                all_values.extend(values)
        
        if not all_values:
            print("  Error: No data found in arrays")
            return False
        
        clim = [np.percentile(all_values, 5), np.percentile(all_values, 95)]
        print(f"  Color limits: {clim}")
        
        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        
        # Process each timestep
        for idx, (time_str, array_name) in enumerate(field_arrays):
            try:
                # Clear previous actors
                plotter.clear()
                
                # Create tube from centerline
                tube = centerline.tube(radius=line_width)
                
                # Add mesh with scalar field
                plotter.add_mesh(tube, scalars=array_name, clim=clim,
                               cmap='viridis', show_edges=False)
                
                # Add title
                time_padded = pad_timestep(time_str)
                plotter.add_text(f"Timestep: {time_padded}\nField: {scalar_field} (1D)", 
                               font_size=20, position='upper_left')
                
                # Add colorbar
                plotter.add_scalar_bar(title=scalar_field.capitalize(), 
                                     vertical=True, title_font_size=16)
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                plotter.screenshot(frame_path)
                image_files.append(frame_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"    Processed {idx + 1}/{len(field_arrays)} frames...")
                    
            except Exception as e:
                print(f"    Warning: Failed to process timestep {time_str}: {e}")
                continue
        
        plotter.close()
        
        if not image_files:
            print("  Error: No frames generated")
            return False
        
        # Create movie from frames
        print(f"  Creating movie from {len(image_files)} frames...")
        clip = ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(output_path, fps=fps, codec='libx264', 
                           bitrate='8000k', audio=False)
        clip.close()
        
        print(f"  Movie saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error creating 1D movie: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Create movies from 3D VTU or 1D centerline VTP solution files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 3D pressure movie from VTU files
  python create_solution_movies.py --3d /path/to/48-procs --output pressure_3d.mp4 --field pressure

  # Create 1D velocity movie from centerline
  python create_solution_movies.py --1d /path/to/unsteady_soln.vtp --output velocity_1d.mp4 --field velocity

  # Create 3D movie with custom slice
  python create_solution_movies.py --3d /path/to/48-procs --output movie.mp4 --slice-normal 0 1 0
        """
    )
    
    # Input type
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--3d', '--3D', dest='input_3d', metavar='DIR',
                            help='Directory containing 3D VTU files (result_*.vtu)')
    input_group.add_argument('--1d', '--1D', dest='input_1d', metavar='FILE',
                            help='1D centerline VTP file with timestep arrays')
    
    # Output
    parser.add_argument('--output', '-o', required=True,
                       help='Output movie file path (.mp4)')
    
    # Field selection
    parser.add_argument('--field', '-f', default='pressure',
                       choices=['pressure', 'velocity'],
                       help='Field to visualize (default: pressure)')
    
    # 3D-specific options
    parser.add_argument('--slice-normal', nargs=3, type=float, default=[1, 0, 0],
                       metavar=('X', 'Y', 'Z'),
                       help='Slice normal vector for 3D visualization (default: 1 0 0)')
    parser.add_argument('--show-mesh', action='store_true',
                       help='Show full 3D mesh instead of slice')
    parser.add_argument('--clim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Color limits [min, max] (auto if not specified)')
    
    # 1D-specific options
    parser.add_argument('--line-width', type=float, default=5.0,
                       help='Centerline tube width for 1D visualization (default: 5.0)')
    
    # General options
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--resolution', nargs=2, type=int, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Output resolution (default: 1920 1080)')
    parser.add_argument('--pattern', default='result_*.vtu',
                       help='File pattern for 3D VTU files (default: result_*.vtu)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process based on input type
    if args.input_3d:
        # Find VTU files
        vtu_files = find_timestep_files(args.input_3d, args.pattern)
        if not vtu_files:
            print(f"Error: No VTU files found in {args.input_3d}")
            sys.exit(1)
        
        success = create_3d_movie(
            vtu_files, args.output, args.field,
            slice_normal=args.slice_normal, clim=args.clim,
            fps=args.fps, resolution=tuple(args.resolution),
            show_mesh=args.show_mesh
        )
    else:
        success = create_1d_movie(
            args.input_1d, args.output, args.field,
            fps=args.fps, resolution=tuple(args.resolution),
            line_width=args.line_width
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

