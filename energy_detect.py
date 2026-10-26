import zmq
import numpy as np
from collections import deque

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "")

levels = np.array([-1.0, -0.333, 0.333, 1.0])
grid = np.array([r + 1j * i for r in levels for i in levels])

disp_history = deque(maxlen=8)
velocity_buffer = deque(maxlen=20)

calib_buffer = []
noise_floor = None
prev_disp = None

print("[*] Calibrating ambient RF jitter floor (remain still)...")

while True:
    try:
        raw = sock.recv()
        symbols = np.frombuffer(raw, dtype=np.complex64)
        if len(symbols) < 48:
            continue

        # Constellation dispersion calculation
        diffs = np.abs(symbols[:, None] - grid[None, :])
        frame_disp = float(np.mean(np.min(diffs, axis=1) ** 2))
        disp_history.append(frame_disp)

        if len(disp_history) < 8:
            continue

        smoothed = float(np.mean(disp_history))

        if prev_disp is None:
            prev_disp = smoothed
            continue

        instant_velocity = abs(smoothed - prev_disp) * 1000.0
        prev_disp = smoothed

        velocity_buffer.append(instant_velocity)
        if len(velocity_buffer) < 20:
            continue

        motion_energy = float(np.mean(velocity_buffer))

        # Auto-calibrate stationary noise floor
        if noise_floor is None:
            calib_buffer.append(motion_energy)
            if len(calib_buffer) >= 40:
                noise_floor = float(np.mean(calib_buffer))
                trigger_thresh = noise_floor * 2.2
                print(f"[+] Resting jitter floor locked: {noise_floor:.2f}")
                print(f"[+] Motion trigger set to: > {trigger_thresh:.2f}")
                print("[*] Active. Step or wave across the path.")
            continue

        trigger_thresh = noise_floor * 2.2

        if motion_energy > trigger_thresh:
            bar = "#" * min(int((motion_energy - trigger_thresh) * 4) + 1, 35)
            print(f"[!] MOTION DETECTED | Energy: {motion_energy:5.2f} (Base: {noise_floor:.2f}) | {bar}")
        else:
            print(f"[.] QUIET           | Energy: {motion_energy:5.2f} (Base: {noise_floor:.2f})", end="\r")

    except KeyboardInterrupt:
        break
