# Optimized and Readable Combined Code for Underwater Communication with Reed-Solomon Error Correction

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from tqdm import tqdm
import logging
import time

# Logging Configuration
logging.basicConfig(filename='combined_results.log', filemode='w', level=logging.DEBUG)

# Reed-Solomon Class
class ReedSolomon:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.t = (n - k) // 2
        self.gf = self._create_galois_field(256)

    def _create_galois_field(self, size):
        return [i for i in range(size)]

    def encode(self, message):
        message = np.array(message, dtype=int).flatten()
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k}.")
        codeword = np.concatenate([message, np.zeros(self.n - self.k, dtype=int)])
        return codeword  # Simplified for demonstration

    def decode(self, received):
        received = np.array(received, dtype=int)
        return received[:self.k]

# Digital Modulator Class
class DigitalModulator:
    def __init__(self, modulation_type='BPSK', f1=1000, f0=2000, sample_rate=10000, bit_time=10):
        self.modulation_type = modulation_type
        self.f1 = f1
        self.f0 = f0
        self.sample_rate = sample_rate
        self.bit_time = bit_time

    def modulate(self, bitstream):
        if self.modulation_type == 'BPSK':
            return np.array([1 if bit == 1 else -1 for bit in bitstream])
        elif self.modulation_type == 'FSK':
            return self.fsk_modulate(bitstream)
        elif self.modulation_type == 'QPSK':
            return self.qpsk_modulate(bitstream)
        elif self.modulation_type == '16-QAM':
            return self.qam_modulate(bitstream, 16)
        else:
            raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

    def fsk_modulate(self, bitstream):
        total_duration = len(bitstream) * self.bit_time
        t = np.arange(0, total_duration, 1 / self.sample_rate)
        fsk_signal = np.zeros(len(t))
        samples_per_bit = int(self.bit_time * self.sample_rate)
        waveform_f1 = np.sin(2 * np.pi * self.f1 * t[:samples_per_bit])
        waveform_f0 = np.sin(2 * np.pi * self.f0 * t[:samples_per_bit])
        for i, bit in enumerate(bitstream):
            fsk_signal[i * samples_per_bit:(i + 1) * samples_per_bit] = waveform_f1 if bit == 1 else waveform_f0
        return fsk_signal

    def qpsk_modulate(self, bitstream):
        if len(bitstream) % 2 != 0:
            raise ValueError("Bitstream length must be even for QPSK modulation.")
        symbols = []
        for i in range(0, len(bitstream), 2):
            real = 1 if bitstream[i] == 0 else -1
            imag = 1 if bitstream[i + 1] == 0 else -1
            symbols.append(complex(real, imag))
        return np.array(symbols)

    def qam_modulate(self, bitstream, M):
        k = int(np.log2(M))
        if len(bitstream) % k != 0:
            raise ValueError(f"Bitstream length must be a multiple of {k} for {M}-QAM modulation.")
        num_symbols = len(bitstream) // k
        symbols = []
        for i in range(num_symbols):
            bits = bitstream[i * k:(i + 1) * k]
            real = (2 * bits[0] - 1) * (2 ** 0.5)
            imag = (2 * bits[1] - 1) * (2 ** 0.5)
            symbols.append(complex(real, imag))
        return np.array(symbols)

# Digital Demodulator Class
class DigitalDemodulator:
    def __init__(self, modulation_type='BPSK', f1=1000, f0=2000, sample_rate=10000, bit_time=10):
        self.modulation_type = modulation_type
        self.f1 = f1
        self.f0 = f0
        self.sample_rate = sample_rate
        self.bit_time = bit_time

    def demodulate(self, received_signal):
        if self.modulation_type == 'BPSK':
            return (received_signal > 0).astype(int)
        elif self.modulation_type == 'FSK':
            return self.fsk_demodulate(received_signal)
        elif self.modulation_type == 'QPSK':
            return self.qpsk_demodulate(received_signal)
        elif self.modulation_type == '16-QAM':
            return self.qam_demodulate(received_signal, 16)
        else:
            raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

    def fsk_demodulate(self, received_signal):
        samples_per_bit = int(self.bit_time * self.sample_rate)
        num_bits = len(received_signal) // samples_per_bit
        demodulated_bits = np.zeros(num_bits, dtype=int)
        ref_signal_0 = np.sin(2 * np.pi * self.f0 * np.arange(0, self.bit_time, 1 / self.sample_rate))
        ref_signal_1 = np.sin(2 * np.pi * self.f1 * np.arange(0, self.bit_time, 1 / self.sample_rate))
        for i in range(num_bits):
            segment = received_signal[i * samples_per_bit:(i + 1) * samples_per_bit]
            corr_0 = np.max(correlate(segment, ref_signal_0))
            corr_1 = np.max(correlate(segment, ref_signal_1))
            demodulated_bits[i] = 0 if corr_0 > corr_1 else 1
        return demodulated_bits

    def qpsk_demodulate(self, received_signal):
        demodulated_bits = []
        for symbol in received_signal:
            real = 0 if np.real(symbol) > 0 else 1
            imag = 0 if np.imag(symbol) > 0 else 1
            demodulated_bits.extend([real, imag])
        return np.array(demodulated_bits)

    def qam_demodulate(self, received_signal, M):
        k = int(np.log2(M))
        demodulated_bits = []
        for symbol in received_signal:
            real = 1 if np.real(symbol) > 0 else 0
            imag = 1 if np.imag(symbol) > 0 else 0
            demodulated_bits.extend([real, imag])
        return np.array(demodulated_bits)

# Underwater Channel Class
class UnderwaterChannel:
    def __init__(self, frequency, absorption_coefficient, scattering_coefficient):
        self.frequency = frequency
        self.absorption_coefficient = absorption_coefficient
        self.scattering_coefficient = scattering_coefficient

    def transmit(self, signal, distance, snr_db):
        total_attenuation = distance * (self.absorption_coefficient + self.scattering_coefficient)
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        return signal * np.exp(-total_attenuation) + noise

# Testing Function

def test_combined_rs(num_nodes):
    rs = ReedSolomon(n=16, k=8)
    modulation_types = ['BPSK', 'QPSK', 'FSK', '16-QAM']
    snr_range = np.linspace(0, 30, 10)
    results = {mod: {'ber': [], 'fer': [], 'success_rate': []} for mod in modulation_types}

    for mod in modulation_types:
        modulator = DigitalModulator(modulation_type=mod)
        demodulator = DigitalDemodulator(modulation_type=mod)
        channel = UnderwaterChannel(frequency=44000, absorption_coefficient=0.1, scattering_coefficient=0.05)

        for snr in snr_range:
            bit_errors = 0
            frame_errors = 0
            total_frames = 100

            for _ in range(total_frames):
                message = np.random.randint(0, 2, rs.k)
                codeword = rs.encode(message)
                modulated_signal = modulator.modulate(codeword)
                received_signal = channel.transmit(modulated_signal, distance=10, snr_db=snr)
                demodulated_signal = demodulator.demodulate(received_signal)
                decoded_message = rs.decode(demodulated_signal)

                bit_errors += np.sum(decoded_message != message)
                frame_errors += 1 if not np.array_equal(decoded_message, message) else 0

            ber = bit_errors / (rs.k * total_frames)
            fer = frame_errors / total_frames
            success_rate = (total_frames - frame_errors) / total_frames

            results[mod]['ber'].append(ber)
            results[mod]['fer'].append(fer)
            results[mod]['success_rate'].append(success_rate)

    # Plot Results
    plt.figure(figsize=(18, 6))

    # BER Plot
    plt.subplot(1, 3, 1)
    for mod in modulation_types:
        plt.plot(snr_range, results[mod]['ber'], label=mod)
    plt.title('Bit Error Rate (BER) vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # FER Plot
    plt.subplot(1, 3, 2)
    for mod in modulation_types:
        plt.plot(snr_range, results[mod]['fer'], label=mod)
    plt.title('Frame Error Rate (FER) vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    # Success Rate Plot
    plt.subplot(1, 3, 3)
    for mod in modulation_types:
        plt.plot(snr_range, results[mod]['success_rate'], label=mod)
    plt.title('Success Rate vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_combined_rs(num_nodes=5)
