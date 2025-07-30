import numpy as np
import matplotlib.pyplot as plt

def Euclid(X, Y):
    """
    Calculates the squared Euclidean distance between two points.
    """
    return (X[0] - Y[0]) ** 2 + (X[1] - Y[1]) ** 2

def WSN_fit(indi, scale=(0, 50), radius=5, gran=1):
    """
    Fitness function for the WSN coverage problem.
    Calculates the coverage ratio of sensors in a given area.
    """
    radius_pow = radius * radius
    lb, ub = scale[0], scale[1]
    # Reshape the 1D individual array into 2D sensor positions (x, y)
    pos = np.array(indi).reshape(-1, 2)
    
    sum_sensor_points = 0
    covered_points = 0
    
    # Iterate over the monitored area grid
    for i in np.arange(lb, ub, gran):
        for j in np.arange(lb, ub, gran):
            sum_sensor_points += 1
            # Check if the grid point is covered by any sensor
            for particle in pos:
                if Euclid(particle, [i, j]) <= radius_pow:
                    covered_points += 1
                    break  # Move to the next grid point once covered
                    
    return covered_points / sum_sensor_points

def Draw_indi(indi, title, name, scale=(0, 50), radius=5):
    """
    Draws the distribution of sensors in the monitoring area.
    """
    plt.figure(figsize=(8, 8))
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[0], scale[1])
    pos = np.array(indi).reshape(-1, 2)

    plt.title(title)
    ax = plt.gca()
    # Draw circles representing sensor coverage
    for particle in pos:
        ax.add_patch(plt.Circle((particle[0], particle[1]), radius=radius, facecolor='skyblue', edgecolor="r", alpha=0.6))
    # Draw points for sensor locations
    for particle in pos:
        ax.scatter(particle[0], particle[1], color='black')
        
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    plt.savefig("./pics/" + name, dpi=300)
    plt.close()

