
import numpy as np
import matplotlib.pyplot as plt

def create_comparison_barchart(results_data):
    """
    Generates and saves a grouped bar chart comparing algorithm performance.

    Args:
        results_data (dict): A dictionary containing the mean coverage for each algorithm.
    """
    algorithms = list(results_data.keys())
    sensor_counts = list(results_data[algorithms[0]].keys())
    
    x = np.arange(len(sensor_counts))  # the label locations
    width = 0.08  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Generate a color for each algorithm
    colors = plt.cm.get_cmap('tab10', len(algorithms))

    # Create the bars for each algorithm
    for i, algo in enumerate(algorithms):
        means = [results_data[algo][sc] for sc in sensor_counts]
        offset = width * (i - len(algorithms) / 2)
        rects = ax.bar(x + offset, means, width, label=algo, color=colors(i))
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Mean Coverage (%)', fontsize=14)
    ax.set_title('WSN Coverage Performance Comparison by Sensor Count', fontsize=16, pad=20)
    ax.set_xticks(x, sensor_counts)
    ax.set_xlabel('Number of Sensors', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Algorithms")
    ax.set_ylim(60, 100) # Set Y-axis to focus on the performance difference
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # Save and show the plot
    output_filename = 'WSN_Algorithm_Comparison_BarChart.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Bar chart comparison saved as '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    # Mean coverage data compiled from Table 10 and our AHADE analysis
    # This data is hardcoded for creating this specific visualization.
    all_results = {
        'PSO': {'32': 71.2, '42': 80.9, '54': 88.4},
        'CMAES': {'32': 67.7, '42': 76.8, '54': 84.9},
        'CSA': {'32': 74.3, '42': 83.9, '54': 90.6},
        'STO': {'32': 69.1, '42': 78.1, '54': 86.2},
        'POA': {'32': 79.1, '42': 87.8, '54': 94.0},
        'SOA': {'32': 69.9, '42': 78.6, '54': 86.2},
        'EVO': {'32': 66.8, '42': 76.0, '54': 84.6},
        'COA': {'32': 77.7, '42': 87.4, '54': 93.5},
        'AHADE': {'32': 80.2, '42': 89.8, '54': 95.3},
        'HRCOA': {'32': 80.0, '42': 89.5, '54': 95.1}
    }
    
    create_comparison_barchart(all_results)
