import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style
sns.set_style("whitegrid")

# Hard-coded data
models = ["SF", "RF", "MLP", "CNN"]
imu_scores = [0.754, 0.784, 0.758, np.nan]
audio_scores = [np.nan, np.nan, np.nan, 0.24]
audio_imu_scores = [0.335, 0.305, 0.315, np.nan]

# Create a bar plot
barWidth = 0.25
r1 = np.arange(len(imu_scores))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, imu_scores, color=sns.color_palette()[0], width=barWidth, edgecolor="white", label="IMU only")
plt.bar(r2, audio_scores, color=sns.color_palette()[1], width=barWidth, edgecolor="white", label="Audio only")
plt.bar(r3, audio_imu_scores, color=sns.color_palette()[2], width=barWidth, edgecolor="white", label="Audio + IMU")

# Add xticks on the middle of the group bars
plt.xlabel("Model", fontweight="bold")
plt.xticks([r + barWidth for r in range(len(imu_scores))], models)

plt.ylim(0, 1)

# Add y-axis label and title
plt.ylabel("Weighted F1 score", fontweight="bold")
plt.title("Comparison of models", fontweight="bold")

# Add legend and show the plot
plt.legend()
# plt.savefig("model_comparison.png", dpi=300)
plt.show()
