"""
Scalable FRQI (Flexible Representation of Quantum Images) Implementation
Supports 2x2, 4x4, and larger image sizes using the proven architecture

This implementation extends the working 2x2 system to support larger images
while maintaining the same high-quality reconstruction performance.

Author: Quantum Image Processing Research
Version: 2.0 (Scalable Release)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import time
from typing import Dict, Tuple, List, Optional

def clean_measurement_string(state_str: str) -> str:
    """Clean and standardize measurement strings from Qiskit output."""
    clean = ''.join(state_str.split())
    # Handle duplicate strings (sometimes Qiskit returns doubled strings)
    if len(clean) % 2 == 0 and clean[:len(clean)//2] == clean[len(clean)//2:]:
        clean = clean[:len(clean)//2]
    return clean

def create_multi_controlled_rotation(qc: QuantumCircuit, theta: float,
                                   controls: List[int], target: int) -> None:
    """
    Create multi-controlled rotation using Qiskit's built-in mcry gate.
    Falls back to manual decomposition for 2-control case if needed.
    """
    if len(controls) == 0:
        qc.ry(theta, target)
    elif len(controls) == 1:
        qc.cry(theta, controls[0], target)
    elif len(controls) == 2:
        # Use the proven 2-control decomposition from the working 2x2 version
        control1, control2 = controls
        qc.ry(theta/2, target)
        qc.cx(control2, target)
        qc.ry(-theta/2, target)
        qc.cx(control1, target)
        qc.ry(theta/2, target)
        qc.cx(control2, target)
        qc.ry(-theta/2, target)
        qc.cx(control1, target)
    else:
        # Use Qiskit's multi-controlled rotation for > 2 controls
        qc.mcry(theta, controls, target)

class ScalableFRQIEncoder:
    """
    Scalable FRQI encoder supporting multiple image sizes.

    This class extends the proven 2x2 implementation to support larger images
    while maintaining the same reconstruction quality and architecture.
    """

    def __init__(self, image_size: int = 2):
        """
        Initialize the scalable FRQI encoder.

        Args:
            image_size: Size of square images (must be power of 2)
        """
        self.image_size = image_size
        self.n_pixels = image_size * image_size
        self.n_position_qubits = int(np.log2(self.n_pixels))
        self.n_color_qubits = 1
        self.n_total_qubits = self.n_position_qubits + self.n_color_qubits

        # Validate input
        if 2**self.n_position_qubits != self.n_pixels:
            raise ValueError(f"Image size {image_size} must be a power of 2")

        if image_size > 8:
            print(f"⚠️ Warning: {image_size}×{image_size} images require {self.n_total_qubits} qubits")
            print(f"   This may be slow on classical simulators")

        print(f"✅ Scalable FRQI Encoder initialized:")
        print(f"   Image size: {image_size}×{image_size} ({self.n_pixels} pixels)")
        print(f"   Position qubits: {self.n_position_qubits}")
        print(f"   Color qubits: {self.n_color_qubits}")
        print(f"   Total qubits: {self.n_total_qubits}")

    def create_sample_image(self, pattern: str = "single") -> np.ndarray:
        """Create sample test images for any size."""
        img = np.zeros((self.image_size, self.image_size))

        if pattern == "single":
            img[0, 0] = 1.0
        elif pattern == "corners":
            img[0, 0] = 1.0
            img[0, -1] = 1.0
            img[-1, 0] = 1.0
            img[-1, -1] = 1.0
        elif pattern == "edge":
            img[0, :] = 1.0
            img[:, 0] = 1.0
        elif pattern == "cross":
            mid = self.image_size // 2
            if self.image_size >= 4:
                img[mid, :] = 1.0
                img[:, mid] = 1.0
            else:
                img[0, 1] = 1.0
                img[1, 0] = 1.0
        elif pattern == "diagonal":
            for i in range(self.image_size):
                img[i, i] = 1.0
        elif pattern == "border":
            img[0, :] = 1.0    # Top
            img[-1, :] = 1.0   # Bottom
            img[:, 0] = 1.0    # Left
            img[:, -1] = 1.0   # Right
        elif pattern == "checkerboard":
            for i in range(self.image_size):
                for j in range(self.image_size):
                    img[i, j] = (i + j) % 2
        elif pattern == "gradient":
            for i in range(self.image_size):
                for j in range(self.image_size):
                    img[i, j] = (i + j) / (2 * (self.image_size - 1))
        elif pattern == "center":
            center = self.image_size // 2
            if self.image_size % 2 == 1:
                img[center, center] = 1.0
            else:
                img[center-1:center+1, center-1:center+1] = 1.0
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return img

    def encode_image(self, image: np.ndarray, verbose: bool = True) -> QuantumCircuit:
        """
        Encode image into quantum circuit using scalable FRQI representation.

        Args:
            image: Input image (values in [0,1])
            verbose: Whether to print encoding details

        Returns:
            Quantum circuit with encoded image
        """
        if image.shape != (self.image_size, self.image_size):
            raise ValueError(f"Image shape {image.shape} doesn't match {(self.image_size, self.image_size)}")

        if verbose:
            print(f"🔧 Encoding {self.image_size}×{self.image_size} image...")
            print(f"   Image range: [{np.min(image):.3f}, {np.max(image):.3f}]")

        # Create circuit with explicit classical register
        qc = QuantumCircuit(self.n_total_qubits, self.n_total_qubits)

        # Step 1: Create superposition of all position states
        for i in range(self.n_position_qubits):
            qc.h(i)

        if verbose:
            print(f"   Created superposition of {2**self.n_position_qubits} position states")

        # Step 2: Encode each non-zero pixel
        flattened_image = image.flatten()
        encoded_pixels = 0

        for pixel_idx in range(self.n_pixels):
            pixel_value = flattened_image[pixel_idx]

            if abs(pixel_value) < 1e-10:
                continue

            # Convert pixel index to binary position
            binary_pos = format(pixel_idx, f'0{self.n_position_qubits}b')
            row, col = pixel_idx // self.image_size, pixel_idx % self.image_size

            if verbose and encoded_pixels < 10:  # Limit output for large images
                print(f"   Encoding pixel {pixel_idx}: ({row},{col}) = {pixel_value:.3f} → pos='{binary_pos}'")

            # Apply controlled rotation for this position
            angle = pixel_value * (np.pi / 2)
            self._encode_pixel_at_position(qc, angle, binary_pos)
            encoded_pixels += 1

        if verbose:
            print(f"   Encoded {encoded_pixels} non-zero pixels")
            print(f"   Circuit depth: {qc.depth()}")
            print(f"   Circuit size: {qc.size()}")

        return qc

    def _encode_pixel_at_position(self, qc: QuantumCircuit, angle: float, binary_pos: str) -> None:
        """Encode a single pixel at the specified binary position."""
        # Determine which position qubits need to be |0⟩ vs |1⟩
        x_gates_applied = []

        for bit_idx, bit_val in enumerate(binary_pos):
            if bit_val == '0':
                qc.x(bit_idx)  # Flip to |1⟩ so we can use as control
                x_gates_applied.append(bit_idx)

        # Apply multi-controlled rotation to color qubit (last qubit)
        control_qubits = list(range(self.n_position_qubits))
        target_qubit = self.n_position_qubits  # Color qubit

        create_multi_controlled_rotation(qc, angle, control_qubits, target_qubit)

        # Undo X gates
        for qubit in x_gates_applied:
            qc.x(qubit)

    def measure_and_reconstruct(self, circuit: QuantumCircuit, shots: int = 2048,
                              verbose: bool = True) -> Tuple[np.ndarray, Dict[str, int], float]:
        """
        Measure quantum circuit and reconstruct the image.

        Returns:
            Tuple of (reconstructed_image, measurement_counts, execution_time)
        """
        if verbose:
            print(f"🔧 Measuring {self.image_size}×{self.image_size} quantum circuit...")

        # Add explicit measurements
        measured_circuit = circuit.copy()
        for i in range(self.n_total_qubits):
            measured_circuit.measure(i, i)

        if verbose:
            print(f"   Shots: {shots}")
            print(f"   Circuit depth: {measured_circuit.depth()}")

        # Execute simulation
        simulator = AerSimulator()
        transpiled_circuit = transpile(measured_circuit, simulator, optimization_level=1)

        start_time = time.time()
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        execution_time = time.time() - start_time

        counts = result.get_counts()

        if verbose:
            print(f"   Execution time: {execution_time:.3f}s")
            print(f"   Measured {len(counts)} unique states")

        # Reconstruct image
        reconstructed_image = self._reconstruct_from_measurements(counts, shots, verbose)

        return reconstructed_image, counts, execution_time

    def _reconstruct_from_measurements(self, counts: Dict[str, int], shots: int,
                                     verbose: bool = True) -> np.ndarray:
        """Reconstruct image from quantum measurement results."""
        reconstructed_image = np.zeros((self.image_size, self.image_size))

        if verbose:
            print(f"   Reconstructing from measurements...")

        # Clean measurement strings
        cleaned_counts = {}
        for state_str, count in counts.items():
            clean_state = clean_measurement_string(state_str)
            if len(clean_state) == self.n_total_qubits:
                cleaned_counts[clean_state] = cleaned_counts.get(clean_state, 0) + count

        successful_reconstructions = 0
        total_signal_probability = 0

        for state, count in cleaned_counts.items():
            # Parse quantum state: Qiskit returns measurements in reverse order
            # Our qubits: [pos0, pos1, ..., posN, color]
            # Measured bits: [color, posN, ..., pos1, pos0]

            color_bit = int(state[0])

            if color_bit == 1:  # Only process states with color bit = 1
                # Extract position bits (reverse order) and convert to position index
                pos_bits_reversed = state[1:]
                pos_bits = pos_bits_reversed[::-1]  # Reverse to correct order

                try:
                    position_idx = int(pos_bits, 2)
                    row = position_idx // self.image_size
                    col = position_idx % self.image_size

                    if 0 <= row < self.image_size and 0 <= col < self.image_size:
                        probability = count / shots

                        # Apply scaling factor based on image size
                        # For larger superpositions, we need higher scaling
                        scale_factor = 2**self.n_position_qubits
                        intensity = min(scale_factor * probability, 1.0)

                        reconstructed_image[row, col] += intensity
                        successful_reconstructions += 1
                        total_signal_probability += probability

                        if verbose and probability > 0.01 and successful_reconstructions <= 5:
                            print(f"     State '{state}' → pos='{pos_bits}' → ({row},{col}) → {intensity:.3f}")

                except (ValueError, IndexError):
                    continue

        if verbose:
            print(f"   Successful reconstructions: {successful_reconstructions}")
            print(f"   Total signal probability: {total_signal_probability:.3f}")
            print(f"   Max reconstructed value: {np.max(reconstructed_image):.3f}")

        return reconstructed_image

    def analyze_and_visualize(self, original: np.ndarray, reconstructed: np.ndarray,
                            counts: Dict[str, int], pattern_name: str = "",
                            execution_time: float = 0) -> Dict[str, float]:
        """Comprehensive analysis and visualization of results."""

        # Calculate metrics
        metrics = self._calculate_metrics(original, reconstructed)

        if execution_time > 0:
            metrics['execution_time'] = execution_time

        # Print analysis
        print(f"\n📊 Analysis for {pattern_name}:")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   Max reconstructed: {metrics['max_reconstructed']:.3f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        print(f"   Normalized overlap: {metrics['normalized_overlap']:.4f}")
        if execution_time > 0:
            print(f"   Execution time: {execution_time:.3f}s")

        # Quality assessment
        if metrics['correlation'] > 0.95 and metrics['max_reconstructed'] > 0.8:
            print(f"   🌟 EXCELLENT quality!")
        elif metrics['correlation'] > 0.8:
            print(f"   ✅ Very good quality!")
        elif metrics['correlation'] > 0.5:
            print(f"   ✅ Good quality!")
        else:
            print(f"   ⚠️ Needs improvement")

        # Create visualization
        self._create_visualization(original, reconstructed, counts, pattern_name, metrics)

        return metrics

    def _calculate_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction quality metrics."""
        metrics = {}

        metrics['mse'] = np.mean((original - reconstructed) ** 2)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['max_error'] = np.max(np.abs(original - reconstructed))
        metrics['max_original'] = np.max(original)
        metrics['max_reconstructed'] = np.max(reconstructed)

        # Correlation
        if np.var(original.flatten()) > 1e-10:
            metrics['correlation'] = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        else:
            metrics['correlation'] = 1.0 if np.var(reconstructed.flatten()) < 1e-10 else 0.0

        # Normalized overlap
        if np.sum(original) > 0:
            metrics['normalized_overlap'] = np.sum(original * reconstructed) / np.sum(original)
        else:
            metrics['normalized_overlap'] = 1.0 if np.sum(reconstructed) == 0 else 0.0

        return metrics

    def _create_visualization(self, original: np.ndarray, reconstructed: np.ndarray,
                            counts: Dict[str, int], pattern_name: str,
                            metrics: Dict[str, float]) -> None:
        """Create comprehensive visualization."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if pattern_name:
            fig.suptitle(f'{pattern_name} - {self.image_size}×{self.image_size} FRQI Processing',
                        fontsize=14, fontweight='bold')

        # Original image
        im1 = axes[0,0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('Original Image')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0,0])

        # Reconstructed image
        max_val = max(np.max(reconstructed), 1.0)
        im2 = axes[0,1].imshow(reconstructed, cmap='gray', vmin=0, vmax=max_val)
        axes[0,1].set_title(f'Quantum Reconstruction\nMax: {metrics["max_reconstructed"]:.3f}')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[0,1])

        # Error visualization
        error_image = np.abs(original - reconstructed)
        im3 = axes[1,0].imshow(error_image, cmap='hot', vmin=0, vmax=np.max(error_image))
        axes[1,0].set_title(f'Absolute Error\nMSE: {metrics["mse"]:.6f}')
        axes[1,0].grid(True, alpha=0.3)
        plt.colorbar(im3, ax=axes[1,0])

        # Measurement statistics (top states only for large images)
        if len(counts) > 0:
            cleaned_counts = {}
            for state_str, count in counts.items():
                clean_state = clean_measurement_string(state_str)
                if len(clean_state) == self.n_total_qubits:
                    cleaned_counts[clean_state] = cleaned_counts.get(clean_state, 0) + count

            # Show top 8 states for readability
            sorted_states = sorted(cleaned_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            states = [s[0] for s in sorted_states]
            values = [s[1] for s in sorted_states]

            # Color code signal states (color bit = 1)
            colors = ['red' if s[0] == '1' else 'steelblue' for s in states]

            axes[1,1].bar(range(len(states)), values, color=colors)
            axes[1,1].set_title('Top Quantum States (red=signal)')
            axes[1,1].set_xlabel('State')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_xticks(range(len(states)))
            axes[1,1].set_xticklabels([s[:8]+'...' if len(s)>8 else s for s in states],
                                    rotation=45, fontsize=8)

        plt.tight_layout()
        plt.show()

def test_scalable_frqi():
    """Test the scalable FRQI implementation with multiple image sizes."""
    print("🚀 TESTING SCALABLE FRQI IMPLEMENTATION")
    print("=" * 60)

    # Test different image sizes
    test_sizes = [2, 4]  # Start with smaller sizes
    test_patterns = ["single", "corners", "cross"]

    all_results = {}

    for size in test_sizes:
        print(f"\n{'='*50}")
        print(f"TESTING {size}×{size} IMAGES")
        print(f"{'='*50}")

        encoder = ScalableFRQIEncoder(image_size=size)
        size_results = {}

        for pattern in test_patterns:
            print(f"\n--- {pattern.upper()} pattern ---")

            # Create test image
            test_image = encoder.create_sample_image(pattern)
            print(f"Test image:\n{test_image}")

            # Encode and measure
            circuit = encoder.encode_image(test_image, verbose=True)
            reconstructed, counts, exec_time = encoder.measure_and_reconstruct(
                circuit, shots=4096, verbose=True
            )

            # Analyze and visualize
            metrics = encoder.analyze_and_visualize(
                test_image, reconstructed, counts,
                f"{size}×{size} {pattern}", exec_time
            )

            size_results[pattern] = {
                'metrics': metrics,
                'success': metrics['correlation'] > 0.7 and metrics['max_reconstructed'] > 0.5
            }

            if size_results[pattern]['success']:
                print(f"   ✅ {pattern} pattern: SUCCESS!")
            else:
                print(f"   ⚠️ {pattern} pattern: Needs improvement")

        all_results[size] = size_results

    # Summary
    print(f"\n🎯 SCALABILITY SUMMARY")
    print(f"=" * 40)

    for size, patterns in all_results.items():
        successful_patterns = sum(1 for p in patterns.values() if p['success'])
        total_patterns = len(patterns)
        print(f"{size}×{size}: {successful_patterns}/{total_patterns} patterns successful")

        for pattern, result in patterns.items():
            status = "✅" if result['success'] else "⚠️"
            corr = result['metrics']['correlation']
            max_val = result['metrics']['max_reconstructed']
            print(f"  {status} {pattern}: correlation={corr:.3f}, max={max_val:.3f}")

    print(f"\n🎉 Scalable FRQI testing complete!")

if __name__ == "__main__":
    test_scalable_frqi()