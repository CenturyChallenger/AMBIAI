import numpy as np
import matplotlib.pyplot as plt

def generate_navigation_plot(mbon_activity_across_headings, target_heading=180):
    """
    Simulates the 'Familiarity Dip' seen in the MB workshop.
    mbon_activity: 1D array of spikes/activity over 360 degrees.
    """
    headings = np.arange(0, 360, 360/len(mbon_activity_across_headings))
    
    plt.figure(figsize=(12, 5))
    
    # 1. Familiarity Curve
    plt.subplot(1, 2, 1)
    plt.plot(headings, mbon_activity_across_headings, color='blue', linewidth=2)
    plt.axvline(target_heading, color='red', linestyle='--', label='Trained Heading')
    plt.xlabel("Heading Direction (Degrees)")
    plt.ylabel("MBON Activity (Spikes)")
    plt.title("Mushroom Body Output: Familiarity Signal")
    plt.legend()
    
    # 2. Polar Plot (Visualizing the 'Search Vector')
    plt.subplot(1, 2, 2, projection='polar')
    theta = np.deg2rad(headings)
    # Invert activity because LOW activity = HIGH familiarity
    familiarity = np.max(mbon_activity_across_headings) - mbon_activity_across_headings
    plt.plot(theta, familiarity, color='green')
    plt.title("Decoded Familiarity (Certainty)")
    
    plt.tight_layout()
    plt.show()

# Usage Example for the Notebook:
headings_sim = np.random.normal(50, 5, 360) # Baseline noise
headings_sim[170:190] -= 30 # Simulate the 'dip' at 180 degrees
generate_navigation_plot(headings_sim)
