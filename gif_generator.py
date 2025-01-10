import numpy as np
import matplotlib.pyplot as plt
import imageio

# Number of frames for the GIF
num_frames = 30
frames = []

# Generate synthetic power consumption data (example)
time = np.linspace(0, 1, 100)
base_power = 10 + 2 * np.sin(2 * np.pi * 10 * time)  # Baseline
nn_activity = np.exp(-50 * (time - 0.5)**2) * 20     # Neural network signature

# Create each frame
for frame in range(num_frames):
    noise = np.random.normal(0, 1, time.shape)  # Adding noise for realism
    gpu_power = base_power + nn_activity * (frame / num_frames) + noise

    # Plot the power data
    plt.figure(figsize=(6, 4))
    plt.plot(time, gpu_power, label="GPU Power Consumption")
    plt.title("Power Side-Channel Attack on Neural Networks")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.ylim(0, 40)
    plt.legend()

    # Highlight the "attack" as a region in the plot
    if frame > num_frames // 3:
        plt.axvspan(0.4, 0.6, color='red', alpha=0.2, label="Attack Region")

    # Add annotations
    if frame == num_frames // 2:
        plt.text(0.5, 25, "Extracting Inference Signature", ha='center', color='red')

    # Save the frame to a temporary file
    filename = f"frame_{frame}.png"
    plt.savefig(filename)
    plt.close()
    frames.append(imageio.imread(filename))

# Save the frames as a GIF
output_gif = "power_side_channel_attack.gif"
imageio.mimsave(output_gif, frames, fps=10)

print(f"GIF saved as {output_gif}")
