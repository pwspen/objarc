import numpy as np
import matplotlib.pyplot as plt
import os

from classes import ArcTask

# This is categorical, not numerical
# So only exact equality matters
# This is what we want for ARC
def fft_cross_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    center=False,          # if True, fftshift outputs so zero-shift is centered
    z=None,                # if not None, return a conservative score: p - z*SE
    remove_background=True # if True, remove color with largest impact (assumed background)
):
    """
    Categorical cross-correlation without wraparound.
    Returns a dict with:
      - matches: number of exact color matches at each shift
      - overlap: number of compared (both-nonempty) pixels at each shift
      - fraction: matches / overlap (NaN where overlap == 0)
      - score (optional): fraction - z * sqrt(p*(1-p)/overlap) (NaN where overlap == 0)
    """
    EMPTY = 0
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    out_h, out_w = h_a + h_b - 1, w_a + w_b - 1

    # Validate colors (1..10 are allowed, 0 is empty)
    unique_colors = np.union1d(np.unique(image_a), np.unique(image_b))
    unique_colors = unique_colors[unique_colors != EMPTY]
    if np.any((unique_colors < 1) | (unique_colors > 10)):
        bad = unique_colors[(unique_colors < 1) | (unique_colors > 10)]
        raise ValueError(f"Color values out of expected range 1-10: {bad}")

    # Accumulate exact matches across colors
    matches = np.zeros((out_h, out_w), dtype=np.float64)
    by_color = {}
    for color in unique_colors:
        mask_a = (image_a == color).astype(np.float64, copy=False)
        mask_b = (image_b == color).astype(np.float64, copy=False)

        a_pad = np.zeros((out_h, out_w), dtype=np.float64)
        b_pad = np.zeros((out_h, out_w), dtype=np.float64)
        a_pad[:h_a, :w_a] = mask_a
        b_pad[:h_b, :w_b] = mask_b

        corr = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

        by_color[color] = corr
    
        matches += corr

    # Remove color with largest impact
    if len(by_color) > 1 and remove_background:
        color_sums = {color: np.sum(corr) for color, corr in by_color.items()}
        worst_color = max(color_sums, key=color_sums.get)
        matches -= by_color[worst_color]

    # Overlap = number of pixels compared (both non-empty)
    nonempty_a = (image_a != EMPTY).astype(np.float64, copy=False)
    nonempty_b = (image_b != EMPTY).astype(np.float64, copy=False)
    a_pad = np.zeros((out_h, out_w), dtype=np.float64); a_pad[:h_a, :w_a] = nonempty_a
    b_pad = np.zeros((out_h, out_w), dtype=np.float64); b_pad[:h_b, :w_b] = nonempty_b
    overlap = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

    # Clean up numerical noise and enforce valid bounds
    overlap = np.clip(np.rint(overlap), 0, None).astype(np.int64)
    matches = np.clip(np.rint(matches), 0, None)
    matches = np.minimum(matches, overlap).astype(np.float64)

    # Fraction of matches (NaN where no pixels were compared)
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = matches / overlap
    fraction[overlap == 0] = np.nan

    result = dict(matches=matches, overlap=overlap, fraction=fraction)

    if z is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            se = np.sqrt(fraction * (1.0 - fraction) / overlap)
            score = fraction - z * se
        score[overlap == 0] = np.nan
        result['score'] = score

    if center:
        for k in list(result.keys()):
            result[k] = np.fft.fftshift(result[k])

    return result

taskname = "45a5af55"
pair_num = 1
task = ArcTask.from_name(taskname)

image_b = task.train_pairs[pair_num].input + 1
image_a = task.train_pairs[pair_num].output + 1
print(image_a)

params = ["matches"]#, "overlap", "score"]


for param in params:
    for nobg in [True, False]:
        # Compute cross-correlation
        corr = fft_cross_correlation(image_a, image_b, z=2, center=True, remove_background=nobg)[param]

        # Compute autocorrelation for comparison
        autocorr_a = fft_cross_correlation(image_a, image_a, z=2, center=True)[param]
        autocorr_b = fft_cross_correlation(image_b, image_b, z=2, center=True)[param]


        # Visualize
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        axes[0, 0].imshow(image_a, cmap='gray')
        axes[0, 0].set_title('Image A')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(image_b, cmap='gray')
        axes[1, 0].set_title('Image B')
        axes[1, 0].axis('off')

        axes[2, 1].imshow(corr, cmap='hot')
        axes[2, 1].set_title('Cross-Correlation')

        axes[0, 1].imshow(autocorr_a, cmap='hot')
        axes[0, 1].set_title('Autocorrelation (Image A)')
        # axes[0, 1].legend()

        axes[1, 1].imshow(autocorr_b, cmap='hot')
        axes[1, 1].set_title('Autocorrelation (Image B)')
        # axes[1, 1].legend()

        axes[2, 0].set_visible(False)

        plt.tight_layout()

        os.makedirs(f"results/{taskname}", exist_ok=True)
        plt.savefig(f"results/{taskname}/correlation_result_{param}_nobg={nobg}.png")