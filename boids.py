from collections import namedtuple
from math import cos, pi, sin

import numpy as np
import pygame


WINDOW_TITLE = 'Boids'
WINDOW_SIZE = 800, 800
FRAME_RATE = 60
N_DIMENSIONS = 2

N_BOIDS = 40
BOID_SIZE = 10
BOID_VELOCITY = 240 / FRAME_RATE

FIELD_OF_VIEW_IN_RADIANS = 0
VISION_RADIUS = 7 * BOID_SIZE
SHOW_VISION_RADI = False

# Steer to avoid crowding local flockmates
SEPARATION = 0.05
# Steer toward average heading of flockmates
ALIGNMENT = 1e-6
# Steer toward average position of flockmates
COHESION = 1e-4

BLACK = 31, 36, 48
WHITE = 255, 255, 255
RED = 182, 100, 71
GREEN = 166, 211, 94
BLUE = 63, 185, 212


Position = namedtuple('position', 'x y')


def counter_clockwise_rotation_matrix(radians: int) -> np.array:
    if not 0 <= radians <= 2 * pi:
        raise ValueError('Rotation amount should be specified in radians')

    return np.array([
        [cos(radians), -sin(radians)],
        [sin(radians),  cos(radians)]
    ])


def main() -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    window = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    # Initialize positions
    positions = np.random.rand(N_BOIDS, N_DIMENSIONS)
    for idx, size in enumerate(WINDOW_SIZE):
        positions[:,idx] *= size

    # Initialize headings
    headings = 0.5 - np.random.rand(N_BOIDS, N_DIMENSIONS)
    headings /= np.linalg.norm(headings, axis=1)[:,None]

    # Position-agnostic transformation matrix
    triangles = BOID_SIZE * np.array([
        np.identity(N_DIMENSIONS),
        counter_clockwise_rotation_matrix(radians=120 * (pi / 180)),
        counter_clockwise_rotation_matrix(radians=240 * (pi / 180)),
    ])

    while pygame.QUIT not in (event.type for event in pygame.event.get()):
        clock.tick(FRAME_RATE)
        window.fill(BLACK)

        # Render boids
        for position, triangle in zip(positions, headings.dot(triangles)):
            points = (position + triangle).astype(int)
            pygame.draw.polygon(window, BLUE, points)
            pygame.draw.circle(window, RED, points[0], 3)
            if SHOW_VISION_RADI:
                pygame.draw.circle(window, WHITE, position.astype(int), VISION_RADIUS, 1)

        # Update headings based on neighboring flockmates
        for idx in range(N_BOIDS):

            neighbors = [
                (other_pos, other_head) for other_idx, (other_pos, other_head)
                in enumerate(zip(positions, headings))
                if other_idx != idx
                and np.linalg.norm(other_pos - positions[idx]) < VISION_RADIUS
            ]

            if not neighbors:
                continue

            flockmate_positions, flockmate_headings = zip(*neighbors)
            flockmate_positions = np.vstack(flockmate_positions)
            flockmate_headings = np.vstack(flockmate_headings)

            # Steer to avoid crowding local flockmates
            headings[idx] = (
                SEPARATION * np.mean(1 / (positions[idx] - flockmate_positions), axis=0)
                + (1 - SEPARATION) * headings[idx]
            )

            # Steer toward the average heading of local flockmates
            headings[idx] = (
                ALIGNMENT * flockmate_headings.mean(axis=0)
                + (1 - ALIGNMENT) * headings[idx]
            )

            # Steer toward the average position of local flockmates
            headings[idx] = (
                COHESION * flockmate_positions.mean(axis=0)
                + (1 - COHESION) * headings[idx]
            )

        # Renormalize the headings to be unit vectors again
        headings /= np.linalg.norm(headings, axis=1)[:,None]

        # Move boids
        positions += BOID_VELOCITY * headings

        # Handle wall collision
        for idx, size in enumerate(WINDOW_SIZE):
            positions[:,idx] %= size

        pygame.display.update()


if __name__ == '__main__':
    main()
