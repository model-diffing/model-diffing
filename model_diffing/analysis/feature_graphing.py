from typing import Any

import bokeh.plotting as bk
import numpy as np
from bokeh.models import HoverTool, Range1d

COLORS = {
    "normal": "#8E8E99",  # off black
    "bottom_x": "#0081A7",  # teal
    "bottom_y": "#009B72",  # green
    "bottom_both": "#D66853",  # reddish
}


def plot_ft_rotations(
    x_vals: np.ndarray[Any, np.dtype[np.float64]],
    y_vals: np.ndarray[Any, np.dtype[np.float64]],
    x_label: str = "Data1",
    y_label: str = "Data2",
    title: str = "Pretty Plot",
    hover_text: dict[int, dict[str, str]] = {},
    output_file: str | None = None,
    percentile_threshold: float = 5,
    value_threshold: float | None = None,
):
    """
    Plot feature rotations using bokeh.
    Args:
        x_vals: The x-values for the plot.
        y_vals: The y-values for the plot.
        hover_text: A dictionary mapping feature indices to hover text (str) or dict of values.
        title: The title of the plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        output_file: The file to save the plot to.
        percentile_threshold: The percentile threshold for colouring features on the axes.
        value_threshold: The value threshold for colouring features on the axes - overrides percentile_threshold.
    """

    if output_file is not None:
        bk.output_file(output_file)
    else:
        # output to current directory
        output_file = "ft_rotations.html"

    scatter_size = 12

    if value_threshold:
        # Calculate thresholds for bottom x
        x_threshold = value_threshold
        y_threshold = value_threshold
    else:
        x_threshold = np.percentile(x_vals, percentile_threshold)
        y_threshold = np.percentile(y_vals, percentile_threshold)

    # Split data into different categories
    indices = range(len(x_vals))

    # Categorize points
    bottom_both = [i for i in indices if x_vals[i] <= x_threshold and y_vals[i] <= y_threshold]
    bottom_x = [i for i in indices if x_vals[i] <= x_threshold and y_vals[i] > y_threshold]
    bottom_y = [i for i in indices if x_vals[i] > x_threshold and y_vals[i] <= y_threshold]
    normal = [i for i in indices if x_vals[i] > x_threshold and y_vals[i] > y_threshold]

    p = bk.figure(title=title, x_axis_label=x_label, y_axis_label=y_label, width=1000, height=800)

    colors = COLORS
    # Plot each category
    categories = {
        "normal": (normal, colors["normal"]),
        "bottom_x": (bottom_x, colors["bottom_x"]),
        "bottom_y": (bottom_y, colors["bottom_y"]),
        "bottom_both": (bottom_both, colors["bottom_both"]),
    }

    for _, (cat_indices, color) in categories.items():
        # Split into hover and non-hover points
        hover_pts = [i for i in cat_indices if i in hover_text]
        non_hover_pts = [i for i in cat_indices if i not in hover_text]

        if hover_pts:
            data = {
                "x": [x_vals[i] for i in hover_pts],
                "y": [y_vals[i] for i in hover_pts],
                "ft_index": [str(i) for i in hover_pts],
            }

            for key in hover_text[0].keys():
                data[key] = [hover_text[i][key] for i in hover_pts]

            hover_source = bk.ColumnDataSource(data=data)

            # Add hover points with solid outline
            hover_renderer = p.scatter(
                "x",
                "y",
                source=hover_source,
                size=scatter_size,
                fill_color=color,
                fill_alpha=1,
                line_color=None,
                # line_color=color,
                # line_alpha=1.0,
                # line_width=1.5)
            )

            tooltips = [("Feature", "@ft_index"), ("X, Y", "@x{0.00}, @y{0.00}")]
            for key in hover_text[0].keys():
                tooltips.append((key, f"@{key}"))

            hover = HoverTool(tooltips=tooltips, renderers=[hover_renderer])
            p.add_tools(hover)

        if non_hover_pts:
            non_hover_source = bk.ColumnDataSource(
                data={"x": [x_vals[i] for i in non_hover_pts], "y": [y_vals[i] for i in non_hover_pts]}
            )
            # Add non-hover points without outline
            p.scatter(
                "x", "y", source=non_hover_source, size=scatter_size, fill_color=color, fill_alpha=0.5, line_color=None
            )

    # Style the plot
    p.grid.grid_line_color = "#E5E5E5"
    p.background_fill_color = None
    p.border_fill_color = None

    # make axes text large and non italic
    p.xaxis.axis_label_text_font_size = "20pt"
    p.xaxis.axis_label_text_font_style = "normal"
    p.yaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_style = "normal"
    # make title large and bold
    p.title.text_font_size = "20pt"
    p.title.text_font_style = "bold"
    # centre title
    p.title.align = "center"
    # fix axes as 0 to 1.05
    p.x_range = Range1d(0, 1.05)
    p.y_range = Range1d(0, 1.05)
    # fix tick marks
    p.xaxis.ticker = np.linspace(0, 1, 11)
    p.yaxis.ticker = np.linspace(0, 1, 11)

    # bk.save(p)
    bk.show(p)


if __name__ == "__main__":
    # Example usage
    # dummy data sampled in range 0-100
    x_vals = np.random.rand(100)
    y_vals = (4 * np.random.rand(100) + x_vals) / 5
    print(len(x_vals))
    # sqrt values to cluster around 1
    x_vals = np.sqrt(x_vals)
    y_vals = np.sqrt(y_vals)

    hover_text = {i: f"Feature {i}" for i in range(100)}
    print(len(hover_text))
    # randomly remove 50% of hover text
    hover_text = {i: hover_text[i] for i in range(100) if np.random.rand() < 0.5}

    title = "Feature Rotations Measured By Cosine Similarity"
    x_label = "Introducing Sleeper Model"
    y_label = "Introducing Sleeper Data"
    output_file = "feature_rotations.html"
    plot_ft_rotations(x_vals, y_vals, hover_text, title, x_label, y_label, output_file)
