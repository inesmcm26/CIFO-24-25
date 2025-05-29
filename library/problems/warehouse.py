"""Problem definition

Description: The challenge of finding the optimal location for a warehouse that minimizes the total delivery cost to a set of customer locations. 

Search space: All possible warehouse locations

Representation: An array with latitude and longitude

Fitness function: f(x)= Distance between warehouse and customer locations weighted by the delivery cost to each customer

Goal: Minimize f(x).
"""

import random
import numpy as np

from library.solution import PSOSolution

class WarehousePSOSolution(PSOSolution):
    def __init__(
        self,
        customer_locations : list[list[float]],
        delivery_cost: list[float],
        repr=None,
    ):
        # Validate repr if it is passed as argument
        if repr:
            repr = self._validate_repr(repr)

        self.customer_locations = customer_locations
        self.delivery_cost = delivery_cost

        super().__init__(repr=repr)

    def _validate_repr(self, repr):
        if isinstance(repr, list) or isinstance(repr, tuple):
            repr = np.array(repr)
        else:
            raise ValueError("Representation must be either a list or a tuple")
        if len(repr) != 2:
            raise ValueError("Representation must contain only two values: latitude and longitude")
        if not all([isinstance(coordinate, float) for coordinate in repr]):
            raise ValueError("Coordinates should be float values")
        return repr

    def fitness(self):
        # Compute the Euclidean distance from the warehouse (self.repr) to each customer
        distances = np.linalg.norm(np.array(self.customer_locations) - self.repr, axis=1)

        # Total delivery cost is the dot product of distances and corresponding delivery costs
        return np.dot(np.array(self.delivery_cost), distances)

    def random_initial_representation(self):
        # Randomly initialize a warehouse location within the bounding box of customer locations
        lats, lons = zip(*self.customer_locations)  # Unzips into two tuples
        # Get min and max bounds for each coordinate
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        # Define custom range margins (alpha and beta) to control the range of random values
        # Since latitude and longitude are so different from each other, let's initilize the latitude between
        # a certain alpha_lat and beta_lat, and the longitude between a certain alpha_lon and beta_lon
        alpha_lat, beta_lat = random.uniform(min_lat, max_lat), random.uniform(min_lat, max_lat)
        alpha_lon, beta_lon = random.uniform(min_lon, max_lon), random.uniform(min_lon, max_lon)

        # Make sure alpha is smaller than beta
        if alpha_lat > beta_lat:
            alpha_lat, beta_lat = beta_lat, alpha_lat
        if alpha_lon > beta_lon:
            alpha_lon, beta_lon = beta_lon, alpha_lon

        # Randomly generate a latitude and longitude within the scaled bounds
        return np.array([random.uniform(alpha_lat, beta_lat), random.uniform(alpha_lon, beta_lon)])