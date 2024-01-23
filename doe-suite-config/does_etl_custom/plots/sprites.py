
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import is_color_like, to_rgba


def make_filled_circle(center, radius, angle=90, fill=0.5, ax=None, fill_color='blue',
                       background_color=None, no_background=False, **kwargs):
    """
    Add a circle that can be filled *ax* (or the current axes).

    If we need to use some transform magic, we can use **kwargs to add special transform args

    :param center position of center of filled circle
    :param radius float radius of circle
    :param angle int angle in degrees for the starting position of the fill. Default (90) is the top
    :param fill float between 0.0, 1.0 indicating how full the circle should be
    :param ax: Matplotlib ax
    :param fill_color Matplotlib color
    :param background_color Matplotlib color or None. When none, it will take a lighter version of the fill_color as background.

    :param kwargs can be used to pass more matplotlib parameters to the matplotlib.patches instances for drawing / transformation.

    """
    assert is_color_like(fill_color), f"fill_color {fill_color} must be a valid matplotlib color!"
    assert 1.0 >= fill >= 0.0, "Fill must be in [0.0, 1.0]!"

    if ax is None:
        ax = plt.gca()
    if background_color is None:
        background_color = to_rgba(fill_color, 0.2)
    if no_background:
        background_color = None

    round_delta = 0.005
    if fill < round_delta:  # special case 1
        w1 = Circle(center, radius, fc=background_color, alpha=0.0 if no_background else 1.0, **kwargs)
        ax.add_artist(w1)
        return w1
    elif fill > 1.0 - round_delta:  # special case 2
        w1 = Circle(center, radius, fc=fill_color, **kwargs)
        ax.add_artist(w1)
        return w1
    else:
        theta1, theta2 = angle, angle + ((1.00 - fill) * 360)
        w1 = Wedge(center, radius, theta1, theta2, fc=background_color, alpha=0.0 if no_background else 1.0, **kwargs)
        w2 = Wedge(center, radius, theta2, theta1, fc=fill_color, **kwargs)
        for wedge in [w1, w2]:
            ax.add_artist(wedge)
        return [w1, w2]