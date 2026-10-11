"""
Visualize Dogs
    This will open up a window displaying randomly selected training
    images for positive and negative labels. Click on the figure to
    refresh with a set of new images. You can save the images using
    the save button. Close the window to break out of the loop.

    Usage: python visualize_labels.py
"""

import matplotlib.pyplot as plt
import numpy as np

from dataset import resize, ImageStandardizer, DogsDataset
from utils import set_random_seed


def main():
    set_random_seed()

    training_set = DogsDataset("train")
    training_set.X = resize(training_set.X)

    positive_idx = np.where(training_set.y == 1)[0]
    negative_idx = np.where(training_set.y == 0)[0]

    standardizer = ImageStandardizer()
    standardizer.fit(training_set.X)

    N = 5

    fig, axes = plt.subplots(nrows=2, ncols=N, figsize=(2 * N, 2 * 2))

    pad = 3
    axes[0, 0].annotate(
        "Positive",
        xy=(0, 0.5),
        xytext=(-axes[0, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[0, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )
    axes[1, 0].annotate(
        "Negative",
        xy=(0, 0.5),
        xytext=(-axes[1, 0].yaxis.labelpad - pad, 0),
        xycoords=axes[1, 0].yaxis.label,
        textcoords="offset points",
        size="large",
        ha="right",
        va="center",
        rotation="vertical",
    )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Generate the images
    while True:
        rand_pos_idx = np.random.choice(positive_idx, size=N, replace=False)
        rand_neg_idx = np.random.choice(negative_idx, size=N, replace=False)

        X_pos = []
        for idx in rand_pos_idx:
            X_pos.append(training_set.X[idx])

        for i, xi in enumerate(X_pos):
            axes[0, i].imshow(xi)

        X_neg = []
        for idx in rand_neg_idx:
            X_neg.append(training_set.X[idx])

        for i, xi in enumerate(X_neg):
            axes[1, i].imshow(xi)

        plt.draw()
        fig = plt.gcf()
        plt.savefig("visualize_labels.png")
        print("Saved figure to visualize_labels.png.")

        def on_close(event):
            exit()

        fig.canvas.mpl_connect("close_event", on_close)

        if plt.waitforbuttonpress():
            break

    plt.close()


if __name__ == "__main__":
    main()
