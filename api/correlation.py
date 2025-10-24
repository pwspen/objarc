import numpy as np
import matplotlib.pyplot as plt
import os

from classes import ArcTask

from utils import fft_cross_correlation

# The functionality of this script has been integrated into the frontend

# This is categorical, not numerical
# So only exact equality matters
# This is what we want for ARC

taskname = "00d62c1b"
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