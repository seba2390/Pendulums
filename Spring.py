import numpy as np


class Spring:
    def __init__(self, start: np.ndarray, end: np.ndarray, num_coils: int = 10, amplitude: float = 1):
        """
        Initialize a Spring object.

        Parameters:
        start (np.ndarray): The starting point of the spring.
        end (np.ndarray): The ending point of the spring.
        num_coils (int): The number of coils in the spring.
        amplitude (float): The amplitude of the "saw-tooth" pattern.
        """
        if np.allclose(start, end):
            raise ValueError("The two points are equal. (x_1,y_1) == (x_2,y_2) ...")
        self.start = np.array(start)
        self.end = np.array(end)
        self.num_coils = num_coils
        self.amplitude = amplitude

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate points for a "saw-tooth" spring shape between two points.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates representing the spring.
        """
        # Calculate the vector from start to end and its length
        vector = self.end - self.start
        length = np.linalg.norm(vector)

        # Unit vector in the direction from start to end
        r_hat = vector / length

        # Perpendicular unit vector to r_hat
        r_hat_perp = np.array([-r_hat[1], r_hat[0]])

        # Calculate the step size between points along the spring
        step_size = length / (2 * self.num_coils + 1)

        # Initialize the list of points with the start point
        points = [self.start.tolist()]

        # Generate each point along the spring
        for step in range(1, 2 * self.num_coils + 1):
            # Calculate the next point along the direction of the spring
            next_point = self.start + step * step_size * r_hat

            # Add or subtract the perpendicular amplitude based on the step
            next_point += self.amplitude / 2 * (-1) ** step * r_hat_perp

            # Append the calculated point to the list
            points.append(next_point.tolist())

        # Add the end point to the list of points
        points.append(self.end.tolist())

        # Separate the x and y coordinates
        x_coords, y_coords = np.array(points)[:, 0], np.array(points)[:, 1]

        return x_coords, y_coords
