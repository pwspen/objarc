import numpy as np
import matplotlib.pyplot as plt

from classes import ArcTask

# This is categorical, not numerical
# So only exact equality matters
# This is what we want for ARC
def fft_cross_correlation(image_a: np.ndarray, image_b: np.ndarray, ignore_colors: tuple[int] = ()) -> np.ndarray:
    EMPTY = 0

    # We wamt to reserve zero for empty, so shift 0-9 -> 1-10
    image_a += 1
    image_b += 1

    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    out_h = h_a + h_b - 1
    out_w = w_a + w_b - 1
    
    ignore_set = set(ignore_colors) | {EMPTY}

    # Get all unique colors from both images
    unique_colors = np.union1d(np.unique(image_a), np.unique(image_b))

    unique_colors = [color for color in unique_colors if color not in ignore_set]

    # Accumulate correlation across all colors
    total_correlation = np.zeros((out_h, out_w))
    
    for color in unique_colors:
        if color < 1 or color > 10:
            raise ValueError(f"Color value {color} out of expected range 0-9")
        # Create binary masks for this color
        mask_a = (image_a == color).astype(float)
        mask_b = (image_b == color).astype(float)
        
        # Pad masks
        a_padded = np.full((out_h, out_w), EMPTY)
        b_padded = np.full((out_h, out_w), EMPTY)
        a_padded[:h_a, :w_a] = mask_a
        b_padded[:h_b, :w_b] = mask_b
        
        # FFT cross-correlation for this color
        fft_a = np.fft.fft2(a_padded)
        fft_b = np.fft.fft2(b_padded)
        product = fft_a * np.conj(fft_b)

        correlation = np.fft.ifft2(product).real
        
        sum = correlation.sum()
        # if sum < 10000:
        total_correlation += correlation
        print(f"Contribution from {color-1}: {sum:.2f}")
    
    # Enhance peaks
    total_correlation = total_correlation ** 2

    return total_correlation

task = ArcTask.from_name("4c3d4a41")

image_b = task.train_pairs[0].input
image_a = task.train_pairs[0].output


# Compute cross-correlation
corr = fft_cross_correlation(image_a, image_b)

# Compute autocorrelation for comparison
auto_corr = fft_cross_correlation(image_a, image_a)

print(f"Image A shape: {image_a.shape}")
print(f"Image B shape: {image_b.shape}")
print(f"Cross-correlation shape: {corr.shape}")
print(f"Autocorrelation shape: {auto_corr.shape}")
print()

# Find peak in autocorrelation (excluding center)
# Center is at (h_a + h_b - 1) // 2
center_y, center_x = (auto_corr.shape[0] // 2, auto_corr.shape[1] // 2)
# Mask out center region to find secondary peaks
masked = auto_corr.copy()
masked[center_y-2:center_y+3, center_x-2:center_x+3] = -np.inf
peak_loc = np.unravel_index(np.argmax(masked), masked.shape)
print(f"Autocorrelation peak (excluding center) at offset: {peak_loc}")
print(f"Distance from center: ({peak_loc[0] - center_y}, {peak_loc[1] - center_x})")
print()

# Find peak in cross-correlation
peak_cross = np.unravel_index(np.argmax(corr), corr.shape)
print(f"Cross-correlation peak at offset: {peak_cross}")
print(f"Correlation value: {corr[peak_cross]}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(image_a, cmap='gray')
axes[0, 0].set_title('Image A')
axes[0, 0].axis('off')

axes[0, 1].imshow(image_b, cmap='gray')
axes[0, 1].set_title('Image B')
axes[0, 1].axis('off')

axes[0, 2].imshow(corr, cmap='hot')
axes[0, 2].set_title('Cross-Correlation')
axes[0, 2].plot(peak_cross[1], peak_cross[0], 'b+', markersize=10)

axes[1, 0].imshow(auto_corr, cmap='hot')
axes[1, 0].set_title('Autocorrelation (Image A)')
axes[1, 0].plot(center_x, center_y, 'b+', markersize=10, label='center')
axes[1, 0].plot(peak_loc[1], peak_loc[0], 'g+', markersize=10, label='peak')
axes[1, 0].legend()

# Log scale for better visibility
axes[1, 1].imshow(np.log1p(corr), cmap='hot')
axes[1, 1].set_title('Cross-Correlation (log scale)')

axes[1, 2].imshow(np.log1p(auto_corr), cmap='hot')
axes[1, 2].set_title('Autocorrelation (log scale)')

plt.tight_layout()
plt.savefig("correlation_results.png")