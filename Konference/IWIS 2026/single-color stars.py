from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORIES = (
    Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\IWIS\IWIS 2026"),
    Path(r"C:\Users\Herman\Python\WE\others-tests\Konference\IWIS 2026"),
)
OUTPUT_FILENAME = "violet_star.svg"

x_positions = np.arange(1, 2)
def create_rainbow_stars() -> None:
    number_of_stars = len(x_positions)
    y_positions = np.zeros(number_of_stars)
    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, 12))

    fig, ax = plt.subplots(figsize=(15, 2.5))
    ax.scatter(
        x_positions,
        y_positions,
        marker="*",
        s=1400,
        c=colors[3],
        edgecolors="black",
        linewidths=1.2,
    )
    ax.set_xlim(min(x_positions) - 1, max(x_positions) + 1)
    ax.set_ylim(-1.4, 1)

    for i, x_position in enumerate(x_positions[2::3]):
        ax.text(
            x_position,
            -0.55,
            r"$f_{\mathrm{" + str(3*(i+1)) + "}}$",
            ha="center",
            va="top",
            color="black",
            fontsize=30,
        )

    plt.axis("off")
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    for output_directory in OUTPUT_DIRECTORIES:
        output_directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_directory / OUTPUT_FILENAME,
            format="svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    plt.show()


if __name__ == "__main__":
    create_rainbow_stars()
