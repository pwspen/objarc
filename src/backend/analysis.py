import numpy as np
from scipy import ndimage

EMPTY_COLOR = 0

def fft_cross_correlation(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> np.ndarray:
    h_a, w_a = image_a.shape
    h_b, w_b = image_b.shape
    out_h, out_w = h_a + h_b - 1, w_a + w_b - 1

    unique_colors = np.union1d(np.unique(image_a), np.unique(image_b))
    unique_colors = unique_colors[unique_colors != EMPTY_COLOR]
    if np.any((unique_colors < 1) | (unique_colors > 10)):
        bad = unique_colors[(unique_colors < 1) | (unique_colors > 10)]
        raise ValueError(f"Color values out of expected range 1-10: {bad}")

    matches = np.zeros((out_h, out_w), dtype=np.float64)
    by_color: dict[int, np.ndarray] = {}
    for color in unique_colors:
        mask_a = (image_a == color).astype(np.float64, copy=False)
        mask_b = (image_b == color).astype(np.float64, copy=False)

        a_pad = np.zeros((out_h, out_w), dtype=np.float64)
        b_pad = np.zeros((out_h, out_w), dtype=np.float64)
        a_pad[:h_a, :w_a] = mask_a
        b_pad[:h_b, :w_b] = mask_b

        corr = np.fft.ifft2(np.fft.fft2(a_pad) * np.conj(np.fft.fft2(b_pad))).real

        by_color[int(color)] = corr
        matches += corr

    return matches

def auto_correlation(image: np.ndarray, center: bool = True) -> np.ndarray:
    corr = fft_cross_correlation(image, image)
    if center:
        corr = np.fft.fftshift(corr)
    return corr

def cross_correlation(image_a: np.ndarray, image_b: np.ndarray, center: bool = True) -> np.ndarray:
    corr = fft_cross_correlation(image_a, image_b)
    if center:
        corr = np.fft.fftshift(corr)
    return corr

def matches(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    template_pixels = np.sum(template != EMPTY_COLOR)
    matches = fft_cross_correlation(grid, template)
    matches = np.where(matches == template_pixels, 1.0, 0.0)
    return matches

def count_matches(grid: np.ndarray, template: np.ndarray) -> int:
    match_map = matches(grid, template)
    return int(np.sum(match_map))

def best_matches(grid: np.ndarray, template: np.ndarray) -> list[tuple[float, int, int]]: # match score, x, y
    template_pixels = np.sum(template != EMPTY_COLOR)
    matches = fft_cross_correlation(grid, template)
    best = np.max(matches)
    positions = np.argwhere(matches == best)
    return [(best / template_pixels, int(x), int(y)) for y, x in positions]

def best_matchscore(grid: np.ndarray, template: np.ndarray) -> float:
    template_pixels = np.sum(template != EMPTY_COLOR)
    matches = fft_cross_correlation(grid, template)
    best = np.max(matches)
    return best / template_pixels

def get_other_d4(img: np.ndarray) -> list[np.ndarray]:
    """Get all D4 transformations of the input image except the original."""
    transforms = []
    for k in range(1, 4):
        transforms.append(np.rot90(img, k=k))
    flipped = np.fliplr(img)
    transforms.append(flipped)
    for k in range(1, 4):
        transforms.append(np.rot90(flipped, k=k))
    return transforms # [90ccw, 180ccw, 270ccw, flip, flip+90ccw, flip+180ccw, flip+270ccw]

def extract_local_maximums(heatmap: np.ndarray, size: int = 3, threshold: float = 0) -> np.ndarray:
    """Extract local maximums from a heatmap using a maximum filter."""
    if size % 2 == 0 or size < 1:
        raise ValueError("Size must be a positive odd integer.")
    filtered = ndimage.maximum_filter(heatmap, size=size, mode="constant")
    local_max = (heatmap == filtered) & (heatmap >= threshold)
    return local_max

def find_regions(img: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be either 4 or 8.")

    # Convert to binary if needed
    if img.ndim == 3:
        binary = (img.any(axis=2) if img.dtype == bool else img.mean(axis=2) > 0).astype(np.uint8)
    else:
        binary = (img > 0).astype(np.uint8)
    
    # Label connected components
    struct = ndimage.generate_binary_structure(2, connectivity // 4)
    labeled, num_features = ndimage.label(binary, structure=struct)
    
    # Crop each region
    regions = []
    for i in range(1, num_features + 1):
        mask = (labeled == i)
        # Get bounding box
        rows, cols = np.where(mask)
        r1, r2 = rows.min(), rows.max() + 1
        c1, c2 = cols.min(), cols.max() + 1
        # Crop the mask (or mask * img to keep color)
        regions.append(mask[r1:r2, c1:c2])
    
    return regions