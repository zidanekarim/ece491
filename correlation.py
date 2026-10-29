#!/usr/bin/env python3
import sys
import time
import select
import threading
import numpy as np
from collections import deque
import uhd

# =====================================================================
# 1. RADIO & SEQUENCE CONFIGURATION
# =====================================================================
CENTER_FREQ = 2.45e9   # 2.45 GHz
SAMPLE_RATE = 5e6      # 5 MS/s (lower rate = zero USB drops/underflows on B210)
TX_GAIN     = 65.0     # TX gain (dB)
RX_GAIN     = 40.0     # RX gain (dB)

N_SEQ       = 1024     # Zadoff-Chu sequence length
ROOT_U      = 25       # Coprime with N_SEQ

n = np.arange(N_SEQ)
tx_seq = (0.7 * np.exp(-1j * np.pi * ROOT_U * n * (n + 1) / N_SEQ)).astype(np.complex64)
TX_FFT_CONJ = np.conj(np.fft.fft(tx_seq))

# 32x tile (32,768 samples) so TX thread has a deep buffer and never underflows
TX_BUFFER = np.tile(tx_seq, 32)


# tx worker
stop_signal = threading.Event()

def tx_worker(tx_streamer):
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False  

    while not stop_signal.is_set():
        tx_streamer.send(TX_BUFFER, md, timeout=1.0)
        md.start_of_burst = False

    md.end_of_burst = True
    tx_streamer.send(np.zeros(1, dtype=np.complex64), md, timeout=0.1)

# USB data chunk size correction: UHD rx_streamer.recv() may return arbitrary chunk sizes, so we need to buffer and pop exact multiples of N_SEQ for correlation.
class ExactFrameStream:
    """Reads exact multiples of N_SEQ samples from UHD rx_streamer without gaps."""
    def __init__(self, rx_streamer, chunk_len):
        self.rx_streamer = rx_streamer
        self.chunk_len = chunk_len
        self.md = uhd.types.RXMetadata()
        self.recv_buf = np.zeros(4096, dtype=np.complex64)
        self.ring = np.zeros(0, dtype=np.complex64)

    def read_aligned_frame(self):
        while len(self.ring) < self.chunk_len:
            n_rx = self.rx_streamer.recv(self.recv_buf, self.md, timeout=0.5)
            if self.md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                # If USB overflowed, sample alignment jumped -> notify caller
                self.ring = np.zeros(0, dtype=np.complex64)
                return None, True
            if n_rx > 0:
                self.ring = np.concatenate((self.ring, self.recv_buf[:n_rx]))

        n_avail_frames = len(self.ring) // self.chunk_len
        frame = self.ring[(n_avail_frames - 1) * self.chunk_len : n_avail_frames * self.chunk_len]
        self.ring = self.ring[n_avail_frames * self.chunk_len :]
        return frame, False

# Channel Impulse Response (CIR) feature extraction: peak power, phase, SNR, and peak bin index
def extract_cir_features(rx_block, ref_bin=None):
    cir = np.fft.ifft(np.fft.fft(rx_block) * TX_FFT_CONJ) / N_SEQ
    cir_pwr = np.abs(cir) ** 2

    if ref_bin is None:
        peak_bin = int(np.argmax(cir_pwr))
    else:
        # Search +/- 4 bins around locked LoS bin
        search_bins = [(ref_bin + k) % N_SEQ for k in range(-4, 5)]
        peak_bin = search_bins[int(np.argmax(cir_pwr[search_bins]))]

    peak_val = cir[peak_bin]
    peak_power = float(np.abs(peak_val) ** 2)
    peak_phase = float(np.angle(peak_val))

    # Mask out main lobe (+/- 3 bins) to compute background/multipath power
    mask = np.ones(N_SEQ, dtype=bool)
    for k in range(-3, 4):
        mask[(peak_bin + k) % N_SEQ] = False

    # Use mean off-peak power in dB (Peak-to-OffPeak Ratio / Inverse SNR)
    mean_offpeak_pwr = float(np.mean(cir_pwr[mask]))
    snr_db = 10.0 * np.log10((peak_power + 1e-15) / (mean_offpeak_pwr + 1e-15))

    return np.array([peak_power, peak_phase, snr_db, peak_bin])


def main():
    print("[*] Initializing USRP B210...")
    usrp = uhd.usrp.MultiUSRP("type=b200")

    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:A"))
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:B"))

    usrp.set_clock_source("internal")
    usrp.set_tx_rate(SAMPLE_RATE, 0)
    usrp.set_rx_rate(SAMPLE_RATE, 0)

    tune_req = uhd.types.TuneRequest(CENTER_FREQ)
    usrp.set_tx_freq(tune_req, 0)
    usrp.set_rx_freq(tune_req, 0)

    usrp.set_rx_agc(False, 0)
    usrp.set_tx_gain(TX_GAIN, 0)
    usrp.set_rx_gain(RX_GAIN, 0)
    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("TX/RX", 0)

    time.sleep(1.0)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)

    # Start continuous TX thread first
    tx_thread = threading.Thread(target=tx_worker, args=(tx_streamer,), daemon=True)
    tx_thread.start()
    time.sleep(0.2)  # Ensure TX is actively radiating before RX starts

    # Start continuous RX stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)

    reader = ExactFrameStream(rx_streamer, N_SEQ)

    history = deque(maxlen=20)
    calib_buffer = []
    baseline = None
    learned_materials = {}
    warmup = 0
    loop_count = 0
    PRINT_INTERVAL = 100

    print("\n[*] B210 Streaming Active.")
    print("[*] Commands: 'c' + Enter = Recalibrate Empty | 'l <name>' + Enter = Learn Material\n")

    try:
        while True:
            rx_frame, overflowed = reader.read_aligned_frame()
            if overflowed:
                if baseline is not None:
                    new_feats = extract_cir_features(reader.read_aligned_frame()[0], ref_bin=None)
                    baseline[3] = new_feats[3]
                continue

            if warmup < 40:
                warmup += 1
                if warmup == 40:
                    print("[*] Hardware settled. Keep zone EMPTY to lock baseline...")
                continue

            ref_bin = int(baseline[3]) if baseline is not None else None
            feats = extract_cir_features(rx_frame, ref_bin=ref_bin)
            history.append(feats)

            if len(history) < 20:
                continue

            h = np.array(history)
            mean_pwr = float(np.mean(h[:, 0]))
            mean_phi = float(np.angle(np.mean(np.exp(1j * h[:, 1]))))
            mean_snr = float(np.mean(h[:, 2]))
            med_bin  = float(np.median(h[:, 3]))

            pwr_db_std = float(np.std(10.0 * np.log10(h[:, 0] + 1e-15)))
            smoothed = np.array([mean_pwr, mean_phi, mean_snr, med_bin])

            if baseline is None:
                calib_buffer.append(smoothed)
                if len(calib_buffer) >= 30:
                    cb = np.array(calib_buffer)
                    baseline = np.array([
                        np.mean(cb[:, 0]),
                        np.angle(np.mean(np.exp(1j * cb[:, 1]))),
                        np.mean(cb[:, 2]),
                        np.round(np.median(cb[:, 3]))
                    ])
                    print(f"[+] Empty Baseline Locked @ Bin {int(baseline[3])} | "
                          f"Empty SNR: {baseline[2]:.1f} dB")
                continue

            # Deltas relative to Empty Air
            dP_dB = 10.0 * np.log10((mean_pwr + 1e-15) / (baseline[0] + 1e-15))
            dPhi_deg = float(np.degrees(np.angle(np.exp(1j * (mean_phi - baseline[1])))))
            # Multipath/scattering drops the Peak-to-Sidelobe SNR (dSNR_dB)
            dSNR_dB = mean_snr - baseline[2]

            delta_vec = np.array([dP_dB, dPhi_deg, dSNR_dB])

            # Check keyboard input (non-blocking, checked every iteration)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd == "c":
                    calib_buffer.clear()
                    baseline = None
                    print("\n[*] Recalibrating Empty Air baseline...")
                    continue
                elif cmd.startswith("l "):
                    name = cmd[2:].strip().upper()
                    learned_materials[name] = delta_vec.copy()
                    print(f"\n[+] Saved '{name}': dP={dP_dB:+.2f}dB, dPhi={dPhi_deg:+.1f}deg, dSNR={dSNR_dB:+.2f}dB")

            loop_count += 1
            if loop_count % PRINT_INTERVAL == 0:
                # State & Material Classification
                if pwr_db_std > 0.35:
                    status = ">>> MOTION / SETTLING <<<"
                elif abs(dP_dB) < 0.4 and abs(dPhi_deg) < 4.0 and abs(dSNR_dB) < 0.5:
                    status = "EMPTY (AIR)"
                elif learned_materials:
                    weights = np.array([1.5, 0.25, 1.0])
                    best_name = min(
                        learned_materials.keys(),
                        key=lambda k: np.linalg.norm(
                            np.array([
                                dP_dB - learned_materials[k][0],
                                np.degrees(np.angle(np.exp(1j * np.radians(dPhi_deg - learned_materials[k][1])))),
                                dSNR_dB - learned_materials[k][2]
                            ]) * weights
                        )
                    )
                    status = f"DETECTED: {best_name}"
                elif dP_dB < -5.0 or dP_dB > 1.2 or abs(dSNR_dB) > 2.5:
                    status = "METAL / CONDUCTOR"
                elif dP_dB < -1.2 and abs(dPhi_deg) > 8.0:
                    status = "WATER / LIQUID"
                elif abs(dP_dB) <= 1.2 and abs(dPhi_deg) > 4.0:
                    status = "PLASTIC / WOOD / DIELECTRIC"
                else:
                    status = "OBJECT PRESENT"

                print(f"Bin:{int(med_bin):4d} | SNR:{mean_snr:4.1f}dB | "
                      f"dP:{dP_dB:+6.2f}dB | dPhi:{dPhi_deg:+6.1f}deg | "
                      f"dSNR:{dSNR_dB:+5.2f}dB | {status}")

    except KeyboardInterrupt:
        print("\n[*] Stopping stream...")
    finally:
        stop_signal.set()
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd)
        tx_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()