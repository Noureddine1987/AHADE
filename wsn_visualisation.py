import os
import numpy as np
import matplotlib.pyplot as plt

def draw_sensor_deployment(ax, title, file_path):
    """
    Draws a sensor deployment from a CSV file onto a Matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to draw on.
        title (str): The title for the subplot.
        file_path (str): The full path to the solution CSV file.
    """
    # Plotting parameters
    scale = (0, 50)
    radius = 5
    
    # Configure subplot appearance
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(scale)
    ax.set_ylim(scale)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_xlabel("X-coordinate", fontsize=10)
    ax.set_ylabel("Y-coordinate", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    if os.path.exists(file_path):
        try:
            # Load actual data from the user's files
            solutions = np.loadtxt(file_path, delimiter=',')
            # Use the first run's solution as a representative best
            pos = solutions[0].reshape(-1, 2)

            # Draw circles for sensor coverage area
            for particle in pos:
                circle = plt.Circle((particle[0], particle[1]), radius, facecolor='lightgreen', edgecolor="black", alpha=0.7)
                ax.add_patch(circle)
            
            # Draw black dots for the actual sensor locations
            for particle in pos:
                ax.scatter(particle[0], particle[1], s=12, color='black', zorder=5)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading:\n{os.path.basename(file_path)}", ha='center', va='center', color='red')
            print(f"Error processing {file_path}: {e}")
    else:
        ax.text(0.5, 0.5, "Data file not found", ha='center', va='center', color='red')
        print(f"Error: Could not find the file at {file_path}")


if __name__ == "__main__":
    # --- Configuration ---
    sol_data_path = os.path.join('AHADE_Data', 'WSN', 'Sol')

    # Data from our previous analysis of AHADE's performance
    results = {
        'AHADE': {'32': 80.2, '42': 89.8, '54': 95.3}
    }
    
    sensor_counts = [32, 42, 54]

    # Create a 1x3 figure to hold the subplots for AHADE only
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('AHADE - Optimal WSN Coverage Deployments', fontsize=18, y=1.0)

    # --- Generate Plots ---
    for i, count in enumerate(sensor_counts):
        file_name = f'WSN_{count}.csv'
        full_path = os.path.join(sol_data_path, file_name)
        
        plot_title = f'{count} Sensors\n(Mean Coverage: {results["AHADE"][str(count)]}%)'
        
        draw_sensor_deployment(axs[i], plot_title, full_path)

    # --- Save and Show the Final Figure ---
    output_filename = 'AHADE_WSN_Deployments.png'
    plt.tight_layout(pad=3.0)
    plt.savefig(output_filename, dpi=300)
    
    print(f"AHADE deployment visualization saved as '{output_filename}'.")
    
    plt.show()

