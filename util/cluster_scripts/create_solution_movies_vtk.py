#!/usr/bin/env python3
"""
Script to generate movies from 3D VTU and 1D centerline VTP solution files using VTK only.
This version doesn't require pyvista or moviepy, only VTK and ffmpeg.
"""

import os
import sys
import glob
import argparse
import tempfile
import shutil
import subprocess

# Try to set up virtual display for headless rendering
# If Xvfb is available, use it; otherwise try off-screen rendering
xvfb_process = None
xvfb_display = None

# Don't set DISPLAY to empty - let setup_virtual_display handle it
# os.environ['DISPLAY'] = ''
os.environ['VTK_USE_OFFSCREEN'] = '1'

try:
    import vtk
    # Force off-screen rendering
    vtk.vtkRenderWindow.GlobalWarningDisplayOff()
    from vtk.util.numpy_support import vtk_to_numpy as v2n
    from vtk.util.numpy_support import numpy_to_vtk as n2v
    
    # Try to use OSMesa if available
    try:
        # Check if OSMesa is available
        vtk.vtkOSOpenGLRenderWindow
        USE_OSMESA = True
    except AttributeError:
        USE_OSMESA = False
        print("Warning: OSMesa not available, will try standard off-screen rendering")
except ImportError:
    print("Error: VTK is required. Make sure VTK is installed and available.")
    sys.exit(1)

import numpy as np

def pad_timestep(time_str, width=5):
    """Pad timestep string with zeros."""
    try:
        return str(int(time_str)).zfill(width)
    except (ValueError, TypeError):
        return time_str

def find_timestep_files(directory, pattern="result_*.vtu"):
    """Find all timestep files in directory."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return []
    
    timesteps = []
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith('result_') and basename.endswith('.vtu'):
            time_str = basename.replace('result_', '').replace('.vtu', '')
            if time_str.isdigit() or (time_str and all(c.isdigit() for c in time_str)):
                timesteps.append((time_str, f))
    
    timesteps.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
    return timesteps

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_xvfb():
    """Check if Xvfb is available."""
    try:
        subprocess.run(['which', 'Xvfb'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_virtual_display():
    """Set up a virtual X display using Xvfb if available."""
    if not check_xvfb():
        print("  Warning: Xvfb not found.")
        print("  Options:")
        print("    1. Load X11 module: module load x11")
        print("    2. Install Xvfb: yum install xorg-x11-server-Xvfb")
        print("    3. Manually start Xvfb: Xvfb :99 -screen 0 1920x1080x24 -ac &")
        print("       Then set: export DISPLAY=:99")
        return None, None
    
    # Try to start Xvfb on a display number
    # Use a random display number to avoid conflicts
    import random
    import time
    
    for attempt in range(20):  # Try more times
        display_num = random.randint(10, 99)
        display = f':{display_num}'
        # Start Xvfb with more options for better compatibility
        xvfb_cmd = ['Xvfb', display, '-screen', '0', '1920x1080x24', '-ac', 
                   '+extension', 'GLX', '+extension', 'RANDR', '+extension', 'RENDER',
                   '-nolisten', 'tcp']
        try:
            # Use a log file to capture errors
            log_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
            log_file.close()
            
            xvfb_process = subprocess.Popen(xvfb_cmd, stdout=open(log_file.name, 'w'), 
                                          stderr=subprocess.STDOUT)
            # Give it more time to start
            time.sleep(1.0)
            # Check if it's still running (if it failed, it would have exited)
            if xvfb_process.poll() is None:
                # Test if display is actually working
                test_cmd = ['xdpyinfo', '-display', display]
                test_result = subprocess.run(test_cmd, capture_output=True, timeout=2)
                if test_result.returncode == 0:
                    os.environ['DISPLAY'] = display
                    os.unlink(log_file.name)  # Clean up log file
                    return xvfb_process, display
                else:
                    # Display not working, kill and try next
                    xvfb_process.terminate()
                    xvfb_process.wait()
            else:
                # Process died, check log
                with open(log_file.name, 'r') as f:
                    log_content = f.read()
                    if log_content:
                        print(f"    Xvfb error on {display}: {log_content[:200]}")
            os.unlink(log_file.name)
        except Exception as e:
            if 'log_file' in locals():
                try:
                    os.unlink(log_file.name)
                except:
                    pass
            continue
    
    print("  Warning: Could not start Xvfb on any available display")
    print("  Try manually: Xvfb :99 -screen 0 1920x1080x24 -ac &")
    print("  Then: export DISPLAY=:99")
    return None, None

def create_3d_movie_vtk(vtu_files, output_path, scalar_field="pressure",
                       slice_normal=[1, 0, 0], clim=None, fps=10,
                       resolution=(1920, 1080), show_mesh=False):
    """Create movie from 3D VTU files using VTK."""
    global xvfb_process, xvfb_display
    
    if not vtu_files:
        print("  No VTU files found")
        return False
    
    if not check_ffmpeg():
        print("  Error: ffmpeg is required. Install with: module load ffmpeg or install ffmpeg")
        return False
    
    # Set up virtual display if needed
    if not os.environ.get('DISPLAY'):
        print("  Setting up virtual X display...")
        xvfb_process, xvfb_display = setup_virtual_display()
        if xvfb_process:
            print(f"  Using virtual display: {xvfb_display}")
        else:
            print("  Warning: No X display available. VTK may fail.")
    
    print(f"  Creating 3D movie from {len(vtu_files)} timesteps...")
    print(f"  Field: {scalar_field}, Output: {output_path}")
    
    temp_dir = tempfile.mkdtemp(prefix="movie_frames_")
    
    try:
        # Read first file to determine color limits
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(vtu_files[0][1])
        reader.Update()
        mesh = reader.GetOutput()
        
        # Determine color limits
        if clim is None:
            all_values = []
            for time_str, vtu_file in vtu_files[:min(10, len(vtu_files))]:
                try:
                    reader.SetFileName(vtu_file)
                    reader.Update()
                    temp_mesh = reader.GetOutput()
                    point_data = temp_mesh.GetPointData()
                    
                    # Find field array
                    field_array = None
                    for i in range(point_data.GetNumberOfArrays()):
                        name = point_data.GetArrayName(i)
                        if name and scalar_field.lower() in name.lower():
                            field_array = point_data.GetArray(i)
                            break
                    
                    if field_array:
                        values = v2n(field_array)
                        if values.ndim == 1:
                            all_values.extend(values)
                        else:
                            all_values.extend(np.linalg.norm(values, axis=1))
                except:
                    continue
            
            if all_values:
                clim = [np.percentile(all_values, 5), np.percentile(all_values, 95)]
            else:
                clim = [0, 1]
        
        print(f"  Color limits: {clim}")
        
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)
        
        # Use OSMesa for off-screen rendering (no X11 required)
        if USE_OSMESA:
            try:
                render_window = vtk.vtkOSOpenGLRenderWindow()
            except:
                render_window = vtk.vtkRenderWindow()
        else:
            render_window = vtk.vtkRenderWindow()
        
        render_window.AddRenderer(renderer)
        render_window.SetSize(resolution[0], resolution[1])
        
        # If we have Xvfb, use it normally; otherwise try off-screen
        if xvfb_process:
            # Use normal rendering with Xvfb - don't set off-screen
            pass
        else:
            render_window.SetOffScreenRendering(1)
            render_window.SetShowWindow(False)
        
        # Initialize the window - wrap in try/except to catch aborts
        try:
            render_window.Initialize()
        except Exception as e:
            print(f"  Error: Render window initialization failed: {e}")
            print("  This might be due to X11/OpenGL issues.")
            print("  Try running with: xvfb-run -a python3 create_solution_movies_vtk.py ...")
            return False
        
        # Process each timestep
        image_files = []
        for idx, (time_str, vtu_file) in enumerate(vtu_files):
            try:
                reader.SetFileName(vtu_file)
                reader.Update()
                mesh = reader.GetOutput()
                
                # Find field array
                point_data = mesh.GetPointData()
                field_array = None
                field_name = None
                for i in range(point_data.GetNumberOfArrays()):
                    name = point_data.GetArrayName(i)
                    if name and scalar_field.lower() in name.lower():
                        field_array = point_data.GetArray(i)
                        field_name = name
                        break
                
                if field_array is None:
                    continue
                
                # Clear previous actors
                renderer.RemoveAllViewProps()
                
                # Create visualization
                if show_mesh:
                    mapper = vtk.vtkDataSetMapper()
                    mapper.SetInputData(mesh)
                    mapper.SetScalarModeToUsePointData()
                    mapper.SelectColorArray(field_name)
                    mapper.SetScalarRange(clim[0], clim[1])
                else:
                    # Create slice
                    plane = vtk.vtkPlane()
                    bounds = mesh.GetBounds()
                    center = [(bounds[0] + bounds[1])/2,
                             (bounds[2] + bounds[3])/2,
                             (bounds[4] + bounds[5])/2]
                    plane.SetOrigin(center)
                    plane.SetNormal(slice_normal)
                    
                    cutter = vtk.vtkCutter()
                    cutter.SetCutFunction(plane)
                    cutter.SetInputData(mesh)
                    cutter.Update()
                    
                    slice_mesh = cutter.GetOutput()
                    if slice_mesh.GetNumberOfPoints() == 0:
                        continue
                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(slice_mesh)
                    mapper.SetScalarModeToUsePointData()
                    mapper.SelectColorArray(field_name)
                    mapper.SetScalarRange(clim[0], clim[1])
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                renderer.AddActor(actor)
                
                # Add colorbar
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(mapper.GetLookupTable())
                scalar_bar.SetTitle(scalar_field.capitalize())
                scalar_bar.SetNumberOfLabels(5)
                renderer.AddActor2D(scalar_bar)
                
                # Add text
                text_actor = vtk.vtkTextActor()
                time_padded = pad_timestep(time_str)
                text_actor.SetInput(f"Timestep: {time_padded}\nField: {scalar_field}")
                text_actor.SetPosition(10, resolution[1] - 60)
                text_actor.GetTextProperty().SetFontSize(20)
                text_actor.GetTextProperty().SetColor(0, 0, 0)
                renderer.AddActor2D(text_actor)
                
                # Render - catch any errors
                try:
                    render_window.Render()
                except Exception as e:
                    print(f"    Warning: Render failed for timestep {time_str}: {e}")
                    continue
                
                # Save frame
                try:
                    window_to_image = vtk.vtkWindowToImageFilter()
                    window_to_image.SetInput(render_window)
                    window_to_image.Update()
                except Exception as e:
                    print(f"    Warning: Failed to capture frame for timestep {time_str}: {e}")
                    continue
                
                writer = vtk.vtkPNGWriter()
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                writer.SetFileName(frame_path)
                writer.SetInputConnection(window_to_image.GetOutputPort())
                writer.Write()
                image_files.append(frame_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"    Processed {idx + 1}/{len(vtu_files)} frames...")
                    
            except Exception as e:
                print(f"    Warning: Failed to process {vtu_file}: {e}")
                continue
        
        render_window.Finalize()
        
        if not image_files:
            print("  Error: No frames generated")
            return False
        
        # Create movie using ffmpeg
        print(f"  Creating movie from {len(image_files)} frames using ffmpeg...")
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-b:v', '8000k',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error creating movie: {result.stderr}")
            return False
        
        print(f"  Movie saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error creating 3D movie: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Clean up Xvfb if we started it
        if xvfb_process:
            xvfb_process.terminate()
            xvfb_process.wait()

def create_1d_movie_vtk(vtp_file, output_path, scalar_field="pressure",
                        fps=10, resolution=(1920, 1080), line_width=5.0):
    """Create movie from 1D centerline VTP file using VTK."""
    global xvfb_process, xvfb_display
    
    if not os.path.exists(vtp_file):
        print(f"  Error: VTP file not found: {vtp_file}")
        return False
    
    if not check_ffmpeg():
        print("  Error: ffmpeg is required. Install with: module load ffmpeg or install ffmpeg")
        return False
    
    # Set up virtual display if needed
    if not os.environ.get('DISPLAY'):
        print("  Setting up virtual X display...")
        xvfb_process, xvfb_display = setup_virtual_display()
        if xvfb_process:
            print(f"  Using virtual display: {xvfb_display}")
        else:
            print("  Warning: No X display available. VTK may fail.")
    
    print(f"  Creating 1D movie from: {vtp_file}")
    print(f"  Field: {scalar_field}, Output: {output_path}")
    
    temp_dir = tempfile.mkdtemp(prefix="movie_frames_")
    
    try:
        # Read centerline
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_file)
        reader.Update()
        centerline = reader.GetOutput()
        
        # Find all timestep arrays
        point_data = centerline.GetPointData()
        field_arrays = []
        for i in range(point_data.GetNumberOfArrays()):
            name = point_data.GetArrayName(i)
            if name and scalar_field.lower() in name.lower() and '_' in name:
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        time_str = parts[-1]
                        int(time_str)
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
            array = point_data.GetArray(array_name)
            if array:
                values = v2n(array)
                all_values.extend(values)
        
        if not all_values:
            print("  Error: No data found in arrays")
            return False
        
        clim = [np.percentile(all_values, 5), np.percentile(all_values, 95)]
        print(f"  Color limits: {clim}")
        
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)
        
        # Use OSMesa for off-screen rendering (no X11 required)
        if USE_OSMESA:
            try:
                render_window = vtk.vtkOSOpenGLRenderWindow()
            except:
                render_window = vtk.vtkRenderWindow()
        else:
            render_window = vtk.vtkRenderWindow()
        
        render_window.AddRenderer(renderer)
        render_window.SetSize(resolution[0], resolution[1])
        
        # If we have Xvfb, use it normally; otherwise try off-screen
        if xvfb_process:
            # Use normal rendering with Xvfb - don't set off-screen
            pass
        else:
            render_window.SetOffScreenRendering(1)
            render_window.SetShowWindow(False)
        
        # Initialize the window - wrap in try/except to catch aborts
        try:
            render_window.Initialize()
        except Exception as e:
            print(f"  Error: Render window initialization failed: {e}")
            print("  This might be due to X11/OpenGL issues.")
            print("  Try running with: xvfb-run -a python3 create_solution_movies_vtk.py ...")
            return False
        
        # Process each timestep
        image_files = []
        for idx, (time_str, array_name) in enumerate(field_arrays):
            try:
                # Create tube from centerline
                tube_filter = vtk.vtkTubeFilter()
                tube_filter.SetInputData(centerline)
                tube_filter.SetRadius(line_width)
                tube_filter.SetNumberOfSides(20)
                tube_filter.Update()
                
                tube = tube_filter.GetOutput()
                
                # Clear previous actors
                renderer.RemoveAllViewProps()
                
                # Create mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(tube)
                mapper.SetScalarModeToUsePointData()
                mapper.SelectColorArray(array_name)
                mapper.SetScalarRange(clim[0], clim[1])
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                renderer.AddActor(actor)
                
                # Add colorbar
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(mapper.GetLookupTable())
                scalar_bar.SetTitle(scalar_field.capitalize())
                scalar_bar.SetNumberOfLabels(5)
                renderer.AddActor2D(scalar_bar)
                
                # Add text
                text_actor = vtk.vtkTextActor()
                time_padded = pad_timestep(time_str)
                text_actor.SetInput(f"Timestep: {time_padded}\nField: {scalar_field} (1D)")
                text_actor.SetPosition(10, resolution[1] - 60)
                text_actor.GetTextProperty().SetFontSize(20)
                text_actor.GetTextProperty().SetColor(0, 0, 0)
                renderer.AddActor2D(text_actor)
                
                # Render - catch any errors
                try:
                    render_window.Render()
                except Exception as e:
                    print(f"    Warning: Render failed for timestep {time_str}: {e}")
                    continue
                
                # Save frame
                try:
                    window_to_image = vtk.vtkWindowToImageFilter()
                    window_to_image.SetInput(render_window)
                    window_to_image.Update()
                except Exception as e:
                    print(f"    Warning: Failed to capture frame for timestep {time_str}: {e}")
                    continue
                
                writer = vtk.vtkPNGWriter()
                frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
                writer.SetFileName(frame_path)
                writer.SetInputConnection(window_to_image.GetOutputPort())
                writer.Write()
                image_files.append(frame_path)
                
                if (idx + 1) % 10 == 0:
                    print(f"    Processed {idx + 1}/{len(field_arrays)} frames...")
                    
            except Exception as e:
                print(f"    Warning: Failed to process timestep {time_str}: {e}")
                continue
        
        render_window.Finalize()
        
        if not image_files:
            print("  Error: No frames generated")
            return False
        
        # Create movie using ffmpeg
        print(f"  Creating movie from {len(image_files)} frames using ffmpeg...")
        frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-b:v', '8000k',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error creating movie: {result.stderr}")
            return False
        
        print(f"  Movie saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error creating 1D movie: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Clean up Xvfb if we started it
        if xvfb_process:
            xvfb_process.terminate()
            xvfb_process.wait()

def main():
    parser = argparse.ArgumentParser(
        description="Create movies from 3D VTU or 1D centerline VTP solution files (VTK version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: This script requires an X display for VTK rendering. Options:
  1. Use xvfb-run: xvfb-run -a python3 create_solution_movies_vtk.py ...
  2. Load X11 module: module load x11
  3. The script will try to auto-start Xvfb if available

Examples:
  # Using xvfb-run (recommended)
  xvfb-run -a python3 create_solution_movies_vtk.py --1d file.vtp -o movie.mp4

  # 3D movie
  xvfb-run -a python3 create_solution_movies_vtk.py --3d /path/to/48-procs -o movie.mp4 --field pressure
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--3d', '--3D', dest='input_3d', metavar='DIR',
                            help='Directory containing 3D VTU files (result_*.vtu)')
    input_group.add_argument('--1d', '--1D', dest='input_1d', metavar='FILE',
                            help='1D centerline VTP file with timestep arrays')
    
    parser.add_argument('--output', '-o', required=True,
                       help='Output movie file path (.mp4)')
    parser.add_argument('--field', '-f', default='pressure',
                       choices=['pressure', 'velocity'],
                       help='Field to visualize (default: pressure)')
    parser.add_argument('--slice-normal', nargs=3, type=float, default=[1, 0, 0],
                       metavar=('X', 'Y', 'Z'),
                       help='Slice normal vector for 3D visualization (default: 1 0 0)')
    parser.add_argument('--show-mesh', action='store_true',
                       help='Show full 3D mesh instead of slice')
    parser.add_argument('--clim', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Color limits [min, max] (auto if not specified)')
    parser.add_argument('--line-width', type=float, default=5.0,
                       help='Centerline tube width for 1D visualization (default: 5.0)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--resolution', nargs=2, type=int, default=[1920, 1080],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Output resolution (default: 1920 1080)')
    parser.add_argument('--pattern', default='result_*.vtu',
                       help='File pattern for 3D VTU files (default: result_*.vtu)')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.input_3d:
        vtu_files = find_timestep_files(args.input_3d, args.pattern)
        if not vtu_files:
            print(f"Error: No VTU files found in {args.input_3d}")
            sys.exit(1)
        
        success = create_3d_movie_vtk(
            vtu_files, args.output, args.field,
            slice_normal=args.slice_normal, clim=args.clim,
            fps=args.fps, resolution=tuple(args.resolution),
            show_mesh=args.show_mesh
        )
    else:
        success = create_1d_movie_vtk(
            args.input_1d, args.output, args.field,
            fps=args.fps, resolution=tuple(args.resolution),
            line_width=args.line_width
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# To run:
# module load x11
# python3 create_solution_movies_vtk.py \
#   --1d /scratch/users/nrubio/synthetic_junctions_reduced_results/CCO_trees/set_3/tree_006/unsteady_soln.vtp \
#   --output /scratch/users/nrubio/movies/tree_006_pressure_1d.mp4 \
#   --field pressure