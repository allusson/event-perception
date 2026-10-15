# Event Camera Directed Study: Context Handoff

Context for continuing the work in a new chat. Last updated Oct 15, 2026.

## Project

Graduate directed study, "Event camera based Perception Methods," supervised by Dr. Du (she/her) at the University of Michigan. Scope: survey event-camera perception methods and implement a representative subset as notebooks and exploration code, which may later feed the event-camera portion of a course Dr. Du is developing on alternative sensors. Deliverables are code and exploration, not polished teaching material, and the work implements existing methods rather than inventing new ones.

## Hardware and data capture

- Camera: iniVation DAVIS346 (346x260). It outputs events, grayscale APS frames from the same pixels, and IMU data.
- Software: DV (dv-gui), run on a CAEN lab PC from the N:\ drive (IT provided a special zip). Recording uses the `output_file` module in the Structure tab, which writes `.aedat4` files. Its output directory should point to N:\ because the C: drive gets wiped. The grayscale view comes from wiring the camera's `frames` output to a visualizer; it's also the easiest way to focus the lens.
- **Recording gotcha (found Oct 1):** the camera's own `frames` output must be wired to `output_file`. In digit4 it wasn't: the 286 "frames" in `digit4.aedat4` are DV Accumulator output built from the events, not APS sensor frames. Evidence: pixel values quantized at multiples of 38 (0.15 x 255, the Accumulator default), 34% exact zeros, frame differences correlate 0.75 to 0.83 with the signed event image, "exposure" exactly 33,000 µs equal to the frame period, and the file's second frame stream is empty. This is why they looked dim. EDI and EFNet cannot be meaningfully run on them.
- Recordings so far: `digit1.aedat4` and `digit4.aedat4` (hand-drawn digits on paper, camera moved by hand). digit4: 1,640,575 events over 8.06 s (~200k events/s); its frames continue to 9.40 s, past the last event.
- `.aedat4` details (via the `aedat` Python package): events have fields `t` (absolute µs, uint64), `x`, `y`, `on` (bool). Each frame dict includes `t`, `exposure_begin_t`, `exposure_end_t` and `pixels` (260x346 uint8). Timestamps need zeroing, and subtracting uint64 values directly can overflow, so cast to int64 first.
- No time for re-recording right now. A future lab session should record a blurry clip (more light, faster motion, fixed manual exposure, APS frames wired) and a flicker clip, and sanity-load both before leaving.

## Repo: event-perception

Public at github.com/allusson/event-perception. Mac: Python 3.12 venv via uv. CAEN Windows: repo on N:\event-perception, venv at %LOCALAPPDATA%\ep-venv (C:\ may be wiped; network-drive venvs were too slow), git via winget; see docs/caen_setup.md. DV also runs on the Mac (official macOS ARM build). Structure: `evcam/` (io, representations, cmax, deblur, deblur_eval, efnet_arch, frequency, metrics, viz), notebooks 01-05, `docs/` (dv_setup_tutorial.md, caen_setup.md, removed_notes.md, paper_notes.md), `data/` (dsec, reblur, stars, led_data) and `weights/` gitignored. Tests: 22 passed + 1 expected failure (EDI c-criterion on a synthetic edge). Notebook markdown was thinned to bare technical text plus equations; Raphael plans to rewrite in his own voice (removed text kept in docs/removed_notes.md).

**Notebook 03 (deblurring on REBlur, 903 test frames):** blurry 34.31 dB; CMax + Richardson-Lucy 34.88 (every 8th frame); EDI selected c 34.71, fixed c 35.01, oracle c 35.42; EFNet 38.07 (paper 38.12); own SCER gives identical EFNet results. Selected c sits at its lower bound in 90% of frames (the paper's criterion does not transfer). EDI is about 1.1 dB below the EFNet paper's EDI number but above the EMP paper's 34.3. Own-data section is a placeholder until a clip with real APS frames exists. Equations in place: CMax warp/objective, PSF, convolution, RL update, EDI derivation, SCER per pixel, EFNet, PSNR/SSIM.

**Notebook 04 (frequency from events):** LED driven by Analog Discovery 2 square wave, recorded at 10, 50, 500, 1000, 3000 Hz (data/led_data/led_XXXXhz.aedat4). Method: pool ON events in an ROI into an event-rate signal (50 us bins), Hann window, FFT, accept peaks whose local ratio (peak power / median power in +-20% band) >= 60 (set from data: LED 180 to 34,885, noise-only <= 21.7). Block spectrum map (10x10 px tiles) replaced the per-pixel median map, which read low by integer factors because pixels skip cycles. Results: all five detected with ~0% error, including 3000 Hz. Busiest pixel fires on 83% of cycles at 50 Hz, 20% at 500, 6.5% at 1000, 0.8% at 3000: the photoreceptor bandwidth limit, which depends on illumination. Loose ends: section 1 equation still says plain argmax; local ratio fragile at low frequency (needs a minimum band width in Hz); "fired" fraction can exceed 100% at 10 Hz.

**Notebook 05 (stars, Chin et al. CVPRW 2019 data, archived copy; Stellarium on a screen, DAVIS 240C, 4 deg/s):** contrast no warp 6.49, translation CMax 11.84, rotation CMax (3D omega, Nelder-Mead) 19.12, GT 19.09. Rotation estimate 3.91 vs 4.00 deg/s, axis error 0.27 deg; per-event velocity error 5.17 px/s (translation) vs 0.15 (rotation).

## Current focus (from the Sept 24 meeting with Dr. Du)

1. **Motion deblurring:** compare warp-based and learning-based methods.
2. **"What can only event cameras do?"**, made concrete.
3. **A brief DV setup and data collection tutorial.**

Structured light is on hold until the event camera work is done. Other ideas from the meeting (event generation, 3DGS / EventSplat) are not current priorities.

## Deblurring: key understanding

**EDI core idea, per pixel:** a blurry pixel is the average of the latent brightness over the exposure, and events give the brightness at each moment as a ratio to the reference time, ratio = exp(c · net events since the reference). So

    sharp = blurry / average(ratios),   i.e.   L(f) = B / [(1/T) ∫ exp(c E(t)) dt]

There's no motion model; every pixel is solved independently, and the only unknown is the contrast threshold c. It fails when c varies across pixels and time, when changes below threshold fire no events, and when noise events get exponentiated. It also requires a frame that physically integrated light; an event-built frame makes it circular.

**Papers read:**
- **EDI (Pan et al., CVPR 2019):** solves the equation directly, choosing c by minimizing TV minus correlation with an event edge map. No training. Also produces a high frame-rate video.
- **EFNet (Sun et al., ECCV 2022):** supervised. Input is the blurry image plus SCER (a discretized EDI inner sum, 2N = 6 channels accumulated outward from the exposure midpoint). Two-stage U-Net with channel-wise cross-modal attention (image queries, event keys and values) and an event-mask gated connection. Free-form output with no hard-wired division. Needs paired GT. Trained on GoPro (simulated events) and REBlur, with pretrained weights available.
- **EVDI++ (Zhang et al., arXiv 2509.08260):** self-supervised. A tiny network (0.357M parameters) learns the divisor (the double integral) from events only, and the division stays hard-wired. Trained with consistency losses, so it needs no GT. The hard-wired structure is what makes self-supervision possible.
- **EMP (Cho et al., CVPR 2026, flagged by Dr. Du):** unpaired data. The blurry path is supervised by EDI output weighted by a confidence map; the sharp path synthesizes blur via Fourier magnitude modulation with a GAN, then deblurs. REBlur PSNR: supervised EFNet 38.1, EMP 35.1 to 35.5, EDI 34.3, EVDI 33.8. EDI collapses to 16.7 on EventAid.
- **Not yet read:** the ECCV 2024 paper (06299) and the DA_event_deblur repo from the meeting notes.

**Framing:** the physics is in every method; what changes is its role and the supervision needed. EDI is the model itself (no data). EFNet uses it to design its input (needs paired GT). EMP uses it as a teacher (needs unpaired sharp images). EVDI++ builds it into the architecture (needs only blurry video and events).

**Warp-based vs. EDI:** EDI is not warp-based. CMax is the warp-based method: it shifts events, estimates flow, and that flow can drive deconvolution (Wiener / Richardson-Lucy) of the frame. Still open: confirm with Dr. Du what she meant by "warp-based."

**Notebook 03 as a ladder of assumptions:** CMax + RL assumes a motion model (constant flow per patch); EDI drops it and assumes one contrast threshold; EFNet drops hand-built physics and learns from paired data. Report EDI at both the GT-free selected c and the oracle c; the gap shows whether EDI is limited by its model or by choosing c.

## "Only event cameras can do," concretely

Common thread: each event has its own microsecond timestamp, so time becomes a continuous variable. Frame methods ask what the scene looks like at time t; event methods ask when each pixel changed.

- Per-event asynchronous updates (live time surface, per-event Kalman tracking, Scheerlinck complementary filter).
- Motion compensation by warping the data itself (contrast maximization; no cross-frame correspondence).
- Continuous-time trajectory estimation (splines in SLAM/VIO).
- Per-pixel frequency from inter-event intervals (flicker, vibration, blinking-LED active markers; a 30 fps camera can't see above 15 Hz).
- Timing-based active sensing and stereo (laser-scan structured light where the timestamp encodes projector position; stereo by temporal coincidence).
- Brightness from integrated change (EDI, HDR reconstruction, high-frame-rate video).
- Spiking networks on neuromorphic hardware (Loihi).
- Caveat on dense learned methods: voxel grids and SCER quantize time and wait for the window, losing the latency advantage; stream-native methods mostly trail in accuracy but win on latency and power.

## Immediate next steps

1. Small notebook 04 fixes (section 1 equation, minimum band width, fired fraction); notebook 03 cleanup (duplicate PSNR/SCER equations, k used for both event index and kernel).
2. Record own deblurring clips with real APS frames (fidget spinner, coin spinning, etc.) and fill notebook 03's placeholder.
3. New notebook 06 (simulation and reconstruction), modular: one loader per source (iPhone slow-mo video, own DAVIS clips, later HS-ERGB) returning a common frames+events bundle. Sections: own minimal event simulator from video (v2e optional later); synthetic blur triples with known c to test EDI with true c and EFNet; events-only reconstruction (Scheerlinck high-pass filter first, E2VID stretch); real data.
4. Optional: brighter LED or DV bias changes to show the frequency cutoff depends on light.

## Working style

Raphael prefers concise prose over heavy bullet lists, no em dashes, and explanations of the reasoning behind technical choices. Simple concrete numeric examples worked well for the math, as did building from one pixel up.