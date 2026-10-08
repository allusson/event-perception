# DV Setup Tutorial (DAVIS346)

## Installing DV

Download this zip onto the N:\ drive of your CAEN computer and unzip it there:
https://www.dropbox.com/scl/fi/0sc33lryhmwxz76gs47yw/dvgui.zip?rlkey=6uts5a8p9kfdjhud2xocydfbx&st=gpigvq0x&dl=0

Launch the software from the `dvgui.cmd` file.

## Camera setup

1. Plug in the DAVIS346 and check the camera input. In the Structure tab there should
   be a camera/capture module representing the DAVIS. If it is not there, add the
   camera input module (top of the screen).
2. Use the grayscale frame view to focus the camera with the three concentric rings
   on the lens.
3. In the camera config parameters, set the exposure time manually to a fixed value
   and write it down.

## Recording

1. Add a recording module (`output_file`) and connect its inputs to all the outputs
   coming from the camera (events, frames, IMU).
2. Make sure the frames input is connected to the camera's own `frames` output, not
   to an Accumulator module. The Accumulator draws a grayscale-looking image from the
   events, and recording it instead of the real frames makes the frames unusable for
   deblurring.
3. Set the output directory to the N:\ drive. Files saved on C:\ get wiped.
4. Record in short segments, since event data is memory intensive.

## Before leaving the lab

Load each recording once to check it has both events and real camera frames:

    python -c "from evcam.io import load_aedat4; load_aedat4(r'data\clip.aedat4')"

The loader warns if the frames look like Accumulator output.