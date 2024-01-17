import pygame
import numpy as np
import torch
from scipy.signal import convolve2d
import time
import pygame_gui

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Conway's Game of Life")

# Set up Pygame clock
clock = pygame.time.Clock()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up grid
rows, cols = 50, 50
grid = np.random.choice([0, 1], size=(rows, cols))

# Convert numpy array to PyTorch tensor
grid_tensor = torch.tensor(grid, dtype=torch.float32)

# Define convolution kernel for neighbor count
kernel = torch.tensor([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=torch.float32)

# Define cell size
cell_size = min(width // cols, height // rows)

# Initialize Pygame GUI
pygame_gui.init()
manager = pygame_gui.UIManager((width, height))

# Create a button for resetting the grid
reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, 10), (100, 50)),
                                            text='Reset', manager=manager)

def draw_grid():
    screen.fill(BLACK)
    for row in range(rows):
        for col in range(cols):
            color = WHITE if grid[row, col] == 1 else BLACK
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

def update_grid():
    global grid_tensor

    # Use PyTorch convolutions for efficient neighbor counting
    neighbor_count = convolve2d(grid_tensor.numpy(), kernel.numpy(), mode='same', boundary='wrap')
    neighbor_count_tensor = torch.tensor(neighbor_count, dtype=torch.float32)

    # Apply Conway's rules using PyTorch operations
    new_grid = grid_tensor.clone()
    new_grid[(grid_tensor == 1) & ((neighbor_count_tensor < 2) | (neighbor_count_tensor > 3))] = 0
    new_grid[(grid_tensor == 0) & (neighbor_count_tensor == 3)] = 1

    grid_tensor = new_grid
    return new_grid.numpy()

running = True
while running:
    time_delta = clock.tick(60) / 1000.0  # Track time since last frame for GUI

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.USEREVENT and event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == reset_button:
                grid = np.random.choice([0, 1], size=(rows, cols))
                grid_tensor = torch.tensor(grid, dtype=torch.float32)

    draw_grid()
    grid = update_grid()

    manager.process_events(event)
    manager.update(time_delta)
    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
