import zmq
import numpy as np
from collections import deque

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://127.0.0.1:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "")

# ideal 16-QAM coordinates
levels = np.array([-1.0, -0.333, 0.333, 1.0])
grid = np.array([r + 1j * i for r in levels for i in levels])

history = deque(maxlen=10)
warmup_counter = 0
WARMUP_FRAMES = 30

calib_buffer = []
baseline_dispersion = None

print("Waiting for radios")

while True:
    try:
        raw = sock.recv()
        symbols = np.frombuffer(raw, dtype=np.complex64)
        if len(symbols) < 48:
            continue

        if warmup_counter < WARMUP_FRAMES:
            warmup_counter += 1
            if warmup_counter == WARMUP_FRAMES:
                print("Collecting baseline")
            continue

        diffs = np.abs(symbols[:, None] - grid[None, :])
        frame_dispersion = float(np.mean(np.min(diffs, axis=1) ** 2))
        history.append(frame_dispersion)
        """baseline detection algorithm, broadcasts and subtracts ideal points from received symbols. compute distance between symbols and reference points
        """
        if len(history) < 10:
            continue

        smoothed = float(np.mean(history))

        if baseline_dispersion is None:
            calib_buffer.append(smoothed)
            if len(calib_buffer) >= 30:
                baseline_dispersion = float(np.mean(calib_buffer))
                print(f"Baseline locked: {baseline_dispersion:.6f}")
                print("Active. Step or wave into the RF path.")
            continue

        ratio = smoothed / baseline_dispersion # ratio between baseline grid and new grid
        delta_pct = (ratio - 1.0) * 100.0

        status = "IDLE"
        if abs(delta_pct) > 8.0:  # 8% threshold, a bit low for now
            status = ">>> MOTION DETECTED <<<"

        print(f"Disp: {smoothed:.6f} | Ratio: {ratio:.2f}x | Delta: {delta_pct:+.1f}% | {status}")

    except KeyboardInterrupt:
        break
