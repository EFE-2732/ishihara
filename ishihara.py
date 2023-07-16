import numpy as np
import svg
from itertools import chain
import shapely.geometry as sg
import shapely.vectorized as sv

def generate_non_overlapping_dots(num_dots, sizes):
    """Generates a set of random dots, making sure none overlap.

    Args:
        num_dots: The number of dots to attempt to generate per size.
        sizes: The sizes of the dots.

    Returns:
        A NumPy array of shape `[num_dots, 3]`, where each row represents a dot.
        The first two elements of each row represent the coordinates of the dot and
        the third element represents the radius of the dot.
    """
    circles = np.empty((0, 3))
    for radius in sorted(sizes, reverse=True):
        # Generate a random list of coordinates and radii for the dots.
        new_circles = np.hstack((np.random.rand(num_dots, 2), np.ones((num_dots, 1)) * radius))

        # filter out points outside circle
        new_circles = new_circles[np.linalg.norm(new_circles[:, :2]-0.5, axis=1) + new_circles[:, 2] < 0.5]

        circles = np.concatenate((circles, filter_intersections(circles, new_circles)))
    return circles


def filter_intersections(old_circles, new_circles):
    """Filters a list of circles to remove intersections. First among the new circles, then between the new and old circles.
    
    Args:
        old_circles: A NumPy array of shape `[num_old_circles, 3]`, where each row represents a circle.
            The first two elements of each row represent the coordinates of the circle's center and
            the third element represents the radius of the circle.
        new_circles: A NumPy array of shape `[num_new_circles, 3]`, where each row represents a circle.
            The first two elements of each row represent the coordinates of the circle's center and
            the third element represents the radius of the circle.

    Returns:
        A NumPy array of shape `[num_new_circles, 3]`, where each row represents a circle.
            The first two elements of each row represent the coordinates of the circle's center and
            the third element represents the radius of the circle.
    """

    # Check which new circles intersect.
    intersections = check_circle_intersection(new_circles, new_circles)

    # remove self-intersections from intersections // might be redundant
    # intersections[np.diag_indices(len(new_circles))] = False

    # remove one of each pair of intersecting circles, so that we only remove one circle per pair
    intersections[np.triu_indices(len(new_circles))] = False

    # Remove the intersecting circles from the new circles.
    new_circles = new_circles[~np.any(intersections, axis=1)]

    # Check which new circles intersect with the old circles.
    intersections = check_circle_intersection(old_circles, new_circles)

    # Remove the intersecting circles.
    new_circles = new_circles[~np.any(intersections, axis=0)]

    return new_circles

def check_circle_intersection(circles1, circles2):
    """Checks which circles intersect with each other.

    Args:
    circles1: A list of NumPy arrays, where each array represents a circle.
        Each circle array has the following shape: `[cx, cy, r]`, where `cx` and `cy` are
        the coordinates of the circle's center and `r` is the circle's radius.

    circles2: A list of NumPy arrays, where each array represents a circle.
        Each circle array has the following shape: `[cx, cy, r]`, where `cx` and `cy` are
        the coordinates of the circle's center and `r` is the circle's radius.

    Returns:
    A NumPy boolean array matching circles2, where each element `True` if the corresponding circles
    intersect and `False` otherwise.
    """

    # Compute the distance between the centers of each pair of circles.
    center_distance = np.linalg.norm(
    np.expand_dims(circles1[:, :2], axis=1) - np.expand_dims(circles2[:, :2], axis=0), axis=2
    )

    # Compute the sum of the radii of each pair of circles.
    radius_sum = np.expand_dims(circles1[:, 2], axis=1) + np.expand_dims(circles2[:, 2], axis=0)

    # Return whether the circles intersect.
    return center_distance < radius_sum

def project_colors(circles, colors1, colors2, pattern):
    """Projects colors onto the circles.
    
    args:
        circles: A list of NumPy arrays, where each array represents a circle.
            Each circle array has the following shape: `[cx, cy, r]`, where `cx` and `cy` are
            the coordinates of the circle's center and `r` is the circle's radius.
        colors1: A list of colors to project onto the circles outside the pattern.
        colors2: A list of colors to project onto the circles within the pattern.
        pattern: A shapely polygon representing the pattern.

    Returns:
        A NumPy array of shape `[num_circles, 1]`, containing the colors of the circles.
    """
    # Compute which circles are within the pattern.
    within_pattern = sv.contains(pattern, circles[:, 0], circles[:, 1])
    colors = np.where(within_pattern, np.random.choice(colors2), np.random.choice(colors1))
    return colors

def draw_plate(circles, colors, filename="plate.svg"):
    """Draws a plate with the given circles.

    Args:
    circles: A list of NumPy arrays, where each array represents a circle.
        Each circle array has the following shape: `[cx, cy, r]`, where `cx` and `cy` are
        the coordinates of the circle's center and `r` is the circle's radius.
    """

    # Scale the coordinates and radius of the circles to fit in a 500x500 SVG.
    scale = 500
    canvas = svg.SVG(
    width=scale,
    height=scale,
    elements=[
        svg.Circle(
            cx=circle[0] * scale,
            cy=circle[1] * scale,
            r=circle[2] * scale,
            fill=colors[i],
        )
        for i, circle in enumerate(circles)
    ],
    )
    #print(canvas)
    with open(filename, "w") as f:
        f.write(canvas.__str__())
# Generate a set of non-overlapping dots.
dots = generate_non_overlapping_dots(2000, chain([0.01]*10, [0.02] * 5, [0.03] * 4))

def create_polygon(center, radius, num_points, offset=0.2):
  """Creates a simple polygon for testing.

  Args:
    center: The center point of the polygon.
    radius: The radius of the polygon.
    num_points: The number of points on the polygon.
    offset: The offset to add to the angle of each point.

  Returns:
    A polygon
  """

  angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + offset

  points = []
  for angle in angles:
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    points.append((x, y))

  polygon = sg.Polygon(points)

  return polygon

center = (0.5, 0.5)
radius = 0.4
num_points = 3
offset = 0.2

polygon = create_polygon(center, radius, num_points, offset)

# Project colors onto the dots.
colors = project_colors(dots, ["#94AD51"], ["#D96F47"], polygon)

# Draw the dots.
draw_plate(dots, colors)
