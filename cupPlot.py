import matplotlib.pyplot as plt
import numpy as np

# Data
cups = ['Cup 1', 'Cup 2', 'Cup 3', 'Cup 4']
data = {
    'Species 1': [0, 13, 7, 1],
    'Species 2': [0, 0, 0, 418],
    'Species 3': [83, 28, 1, 0]
}

# Convert to numpy array for easier processing
species = list(data.keys())
values = np.array(list(data.values()))

# Histogram-like horizontal bar plot
num_cups = len(cups)
bar_width = 0.8 / len(species)  # Width of individual bars

fig, ax = plt.subplots(figsize=(10, 6))

# Offset positions for each species
for i, species_name in enumerate(species):
    positions = np.arange(num_cups) + i * bar_width
    ax.bar(positions, values[i], width=bar_width, label=species_name)

# Formatting the plot
ax.set_xlabel('Cups')
ax.set_ylabel('Number of Individuals (Log Scale)')
ax.set_title('Bar Plot of Species Across Cups (Log Scale)')
ax.set_yscale('log')  # Apply log scale
ax.set_xticks(np.arange(num_cups) + bar_width * (len(species) - 1) / 2)
ax.set_xticklabels(cups, rotation=45)
ax.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("CupPlot_LogScale.png")
plt.close()
