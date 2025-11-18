import numpy as np
import matplotlib.pyplot as plt


def create_fixed_maze(size=15):
    """
    Create a 15x15 fixed maze layout.

    Returns:
        np.ndarray: 2D array where 0=empty, 1=wall
    """
    # Start with all empty
    maze = np.zeros((size, size), dtype=int)

    # Add outer walls
    maze[0, :] = 1  # Top wall
    maze[-1, :] = 1  # Bottom wall
    maze[:, 0] = 1  # Left wall
    maze[:, -1] = 1  # Right wall

    # Create interior structure - corridors and rooms

    # Vertical walls (creating corridors)
    maze[3:12, 4] = 1  # Left corridor wall
    maze[3:12, 10] = 1  # Right corridor wall

    # Horizontal walls (creating rooms)
    maze[5, 5:10] = 1  # Upper horizontal wall
    maze[9, 5:10] = 1  # Lower horizontal wall

    # Add some obstacles in corridors
    maze[7, 2] = 1
    maze[7, 12] = 1

    # Create doorways (remove some walls for passages)
    maze[5, 7] = 0  # Door in upper wall
    maze[9, 7] = 0  # Door in lower wall
    maze[7, 4] = 0  # Door in left wall
    maze[7, 10] = 0  # Door in right wall

    # Add some internal obstacles
    maze[3, 7] = 1
    maze[11, 7] = 1
    maze[7, 6:9] = 1  # Central obstacle
    maze[7, 7] = 0  # Opening in central obstacle

    # Ensure start (bottom-left area) is clear
    maze[13, 1:3] = 0

    # Ensure goal (top-right corner) is reachable
    maze[1, 13] = 0  # Goal position cleared

    return maze


def visualize_maze(maze):
    """
    Visualize the maze using matplotlib.

    Args:
        maze: 2D numpy array (0=empty, 1=wall)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create color map: white=empty, black=wall
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])

    # Display maze
    im = ax.imshow(maze, cmap=cmap, interpolation='nearest')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Add coordinate labels
    ax.set_xticks(np.arange(maze.shape[1]))
    ax.set_yticks(np.arange(maze.shape[0]))

    # Mark start position (bottom-left area, e.g., 13,1)
    ax.plot(1, 13, 'go', markersize=15, label='Start Area')

    # Mark goal position (top-right, e.g., 1,13)
    ax.plot(13, 1, 'r*', markersize=20, label='Goal')

    ax.set_title('Maze Layout (15x15)', fontsize=16)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('maze_layout.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Maze shape: {maze.shape}")
    print(f"Total walls: {np.sum(maze == 1)}")
    print(f"Empty cells: {np.sum(maze == 0)}")


def test_maze():
    """Test maze generation and visualization"""
    maze = create_fixed_maze(15)

    # Verify maze properties
    assert maze.shape == (15, 15), "Maze should be 15x15"
    assert maze[0, :].all() == 1, "Top wall should be complete"
    assert maze[-1, :].all() == 1, "Bottom wall should be complete"
    assert maze[:, 0].all() == 1, "Left wall should be complete"
    assert maze[:, -1].all() == 1, "Right wall should be complete"

    # Check goal is accessible (not a wall)
    assert maze[1, 13] == 0, "Goal position should be empty"

    print("✓ Maze validation passed!")

    # Visualize
    visualize_maze(maze)


if __name__ == '__main__':
    test_maze()