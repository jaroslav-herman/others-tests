from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wepy.basics as we


DATA_FILE = Path(
    r"C:\Users\Herman\Python\WE\others-tests\Used data for dataset"
) / "conventional_PEIS__1,46V_OCV180s_open_cell_PEIS.mpt"
OUTPUT_DIRECTORIES = (
    Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\IWIS\IWIS 2026"),
    Path(r"C:\Users\Herman\Python\WE\others-tests\Konference\IWIS 2026"),
)
OUTPUT_FILENAME = "nyquist_cycle_3.svg"


def plot_nyquist_cycle() -> None:
    data = we.read_file_safe(str(DATA_FILE), on_error="raise")
    cycle = data[data["cycle number"] == 3].copy()
    cycle = cycle.sort_values("freq/Hz", ascending=False).reset_index(drop=True)

    number_of_points = 12
    point_indices = np.linspace(
        0, len(cycle) - 1, number_of_points, dtype=int
    )
    selected = cycle.iloc[point_indices]
    colors = we.get_colors(number_of_points)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        selected["Re(Z)/Ohm"],
        selected["-Im(Z)/Ohm"],
        marker = '*',
        edgecolors="black",
        c=colors,
        linewidths=0.5,
        s=200,
    )
    ax.set_aspect("equal")

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
    plot_nyquist_cycle()
