# Underwater Communication Simulation with Reed-Solomon Error Correction

This project implements a simulation for underwater communication networks, incorporating Reed-Solomon (RS) error correction and various digital modulation schemes. It evaluates the performance of the system in terms of Bit Error Rate (BER), Frame Error Rate (FER), and Success Rate across different modulation types.

## Features

- **Reed-Solomon Error Correction**: Implements encoding and decoding for error correction.
- **Modulation Schemes**:
  - Binary Phase Shift Keying (BPSK)
  - Quadrature Phase Shift Keying (QPSK)
  - Frequency Shift Keying (FSK)
  - 16-QAM (Quadrature Amplitude Modulation)
- **Underwater Channel Model**: Simulates signal attenuation and noise over varying distances and Signal-to-Noise Ratios (SNR).
- **Performance Metrics**:
  - BER: Fraction of incorrectly decoded bits.
  - FER: Fraction of frames with errors.
  - Success Rate: Percentage of error-free transmissions.
- **Visualization**: Generates plots for BER, FER, and Success Rate against SNR.


## Main File:
main_uw_rs.py

## Dependencies

The following Python libraries are required to run the simulation:

- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`

Install the dependencies using pip:
```bash
pip install numpy matplotlib scipy tqdm
```

## Usage

1. Clone or download the repository.
2. Run the `main_uw_rs.py` Python script:

```bash
python <main_uw_rs.py>.py
```

The script will simulate the communication system and display the performance metrics in graphical form.

## Code Structure

- **`ReedSolomon` Class**:
  - Implements RS encoding and decoding for error correction.
- **`DigitalModulator` Class**:
  - Handles signal modulation for BPSK, QPSK, FSK, and 16-QAM.
- **`DigitalDemodulator` Class**:
  - Demodulates received signals into binary data.
- **`UnderwaterChannel` Class**:
  - Simulates signal attenuation and noise in an underwater environment.
- **`test_combined_rs` Function**:
  - Executes the simulation across multiple nodes, SNR values, and modulation schemes.

## Outputs

The script outputs three performance plots for each modulation scheme:

1. **BER vs. SNR**: Log-scale plot showing the fraction of incorrect bits as SNR increases.
2. **FER vs. SNR**: Log-scale plot showing the fraction of frames with errors.
3. **Success Rate vs. SNR**: Plot showing the percentage of error-free transmissions.

## Example Plots

- **BER vs. SNR**
- **FER vs. SNR**
- **Success Rate vs. SNR**

(Plots will be displayed when the script is executed.)

## Author

Developed by a simulation enthusiast with a focus on communication systems and error correction in underwater networks.

## License

This project is licensed under the MIT License.

