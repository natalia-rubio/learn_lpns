import pyvista as pv
import os
import moviepy.video.io.ImageSequenceClip

image_files = []
#mesh = pv.read("/home/nrubio/Desktop/steady_state_junction_state.pvsm")
#mesh = pv.read("/home/nrubio/Desktop/synthetic_junctions/geom_0/solution_flow_2.vtu")
#mesh = pv.read("/home/nrubio/Desktop/synthetic_junctions/geom_angle_10/solution_flow_11.188888888888888.vtu")
#mesh = pv.read("/home/nrubio/Desktop/synthetic_junctions/aorta_91/solution_flow_45.vtu")
mesh = pv.read("/home/nrubio/Desktop/synthetic_junctions_complete/geom_0/solution_flow_0.vtu")
#cpos = mesh.plot(scalars = "pressure_00001")
# for i in range(100):
#
#     if i < 10:
#         field_num = "00" + str(i)
#     elif i < 100:
#         field_num = "0" + str(i)
#     elif i < 1000:
#         field_num = str(i)
#
#     plotter = pv.Plotter(off_screen = True)
#     slice = mesh.slice(normal=[1, 0, 0])
#     plotter.add_mesh(mesh = slice, clim=[0, 1333*30], below_color='purple', above_color='yellow', scalars = f"pressure_00{field_num}")
#     image_path = f"movie_images/shot_pressure_00{field_num}.png"
#     plotter.screenshot(image_path)
#
#     image_files.append(image_path)
#
# fps=5
#
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile('aorta91_transient.mp4')

for i in range(41):

    if i < 10:
        field_num = "00" + str(i)
    elif i < 100:
        field_num = "0" + str(i)
    elif i < 1000:
        field_num = str(i)

    plotter = pv.Plotter(off_screen = True)
    slice = mesh.slice(normal=[1, 0, 0])
    plotter.add_mesh(mesh = slice, clim=[0, 6*10**3], below_color='purple', above_color='yellow', scalars = f"pressure_00{field_num}")
    image_path = f"/home/nrubio/Desktop/SV_scripts/movie_images/shot_pressure_00{field_num}.png"
    plotter.screenshot(image_path)

    image_files.append(image_path)

fps=5

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('/home/nrubio/Desktop/SV_scripts/geom0_transient_pressure.mp4')
