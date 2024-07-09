from colorspacious import cspace_converter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

__canvas = None
__image = None
converters = {}

_deuter50_space = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 50}
converters["deuter50"] = cspace_converter(_deuter50_space, "sRGB1")
_deuter100_space = {
    "name": "sRGB1+CVD",
    "cvd_type": "deuteranomaly",
    "severity": 100,
}
converters["deuter100"] = cspace_converter(_deuter100_space, "sRGB1")

_prot50_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 50}
converters["prot50"] = cspace_converter(_prot50_space, "sRGB1")

_prot100_space = {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100}
converters["prot100"] = cspace_converter(_prot100_space, "sRGB1")


def get_canvas(fig=None):
    global __canvas
    if fig is None:
        __canvas = FigureCanvas(plt.gcf())
    else:
        return FigureCanvas(fig)


def get_image(canvas=None):
    global __canvas, __image
    if canvas is None:
        if __canvas is None:
            get_canvas()
        __canvas.draw()  # draw the canvas, cache the renderer
        __image = np.array(__canvas.buffer_rgba(), dtype=float)
    else:
        canvas.draw()  # draw the canvas, cache the renderer
        return np.array(canvas.buffer_rgba(), dtype=float)


def show_transformed(image=None):
    global __image, converters
    if image is None:
        image = __image

    for name, converter in converters.items():
        newrgb = np.clip(converter(image[:, :, :3] / 256), 0, 1)
        fig = plt.figure(figsize=(8, 5))
        plt.imshow(newrgb)
        plt.title(name)
        plt.gca().set_axis_off()
        plt.show()
