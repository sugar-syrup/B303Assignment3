import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import pandas as pd
import numpy as np
import os

# Output directory setup
output_dir = "spider_plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/individual", exist_ok=True)

# Read input JSON file
print(f"Reading {sys.argv[1]}...")

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

# Process Greenhouse and Animal House data
greenhouse = pd.DataFrame(data["Greenhouse"])
animal_house = pd.DataFrame(data["Animal House"])

greenhouse.name = "Greenhouse"
animal_house.name = "Animal House"

class Population:
    def __init__(self, name, data: list[pd.DataFrame], scale=1, separate_communities=False):
        self.name = name
        self.data = data
        species = []
        for community in data:
            species.extend(community.columns[1:])
        self.species = np.unique(species)
        self.id_species = {id: species for id, species in enumerate(self.species)}
        self.scale = scale
        self.separate_communities = separate_communities
        self.create_sample_pop()

    def create_sample_pop(self):
        if self.separate_communities:
            sample_pop = []
            for community in self.data:
                community_count = {}
                for species in community.columns[1:]:
                    community_count[species] = community[species].sum() * self.scale
                sample_pop.append(pd.DataFrame(community_count, index=[0]))
            self.sample_pop = sample_pop
        else:
            sample_pop = {}
            for community in self.data:
                for species in community.columns[1:]:
                    if species in sample_pop:
                        sample_pop[species] += community[species].sum() * self.scale
                    else:
                        sample_pop[species] = community[species].sum() * self.scale
            self.sample_pop = pd.DataFrame(sample_pop, index=[0])

    def get_random_obs(self, total_count):
        if self.separate_communities:
            obs = []
            for community in self.sample_pop:
                total_individuals = community.values.sum()
                species_probs = (community.values / total_individuals)[0]
                selected_sample = np.random.choice(
                    community.columns,
                    size=total_count,
                    p=species_probs
                )
                specii, counts = np.unique(selected_sample, return_counts=True)
                counts = dict(zip(specii, counts))
                obs.append(counts)
            return obs
        else:
            total_individuals = self.sample_pop.values.sum()
            species_probs = (self.sample_pop.values / total_individuals)[0]
            selected_sample = np.random.choice(
                self.sample_pop.columns,
                size=total_count,
                p=species_probs
            )
            specii, counts = np.unique(selected_sample, return_counts=True)
            counts = dict(zip(specii, counts))
        return counts

# Initialize Population for spider data
spider_population = Population(
    "spider Population",
    [greenhouse, animal_house],
    separate_communities=True
)

# Plot Number of Species vs Total Individuals
seed = 42
np.random.seed(seed)
trials_per_point = 30
total_individuals_x_axes = np.arange(0, 1000)
num_species_y_axes = []
for total_individuals in tqdm(total_individuals_x_axes):
    num_species = np.zeros(len(spider_population.data))
    for _ in range(trials_per_point):
        obs = spider_population.get_random_obs(total_individuals)
        num_species += [len(x.values()) for x in obs]
    num_species /= trials_per_point
    num_species_y_axes.append(np.copy(num_species))

num_species_y_axes = np.array(num_species_y_axes)

# Plot results
fig, ax = plt.subplots(figsize=(10, 7))
for i, community in enumerate(spider_population.data):
    ax.plot(total_individuals_x_axes, num_species_y_axes[:, i], label=community.name)
fig.suptitle("Number of Species vs Total Individuals", fontsize=14, fontweight='bold')
ax.set_xlabel("Total Individuals", fontsize=12)
ax.set_ylabel("Number of Species", fontsize=12)
ax.legend()
ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.savefig(f"{output_dir}/num_species_vs_total_individuals_spiders.png")
plt.close()

# Calculate relative abundance
def calculate_relative_abundance(data):
    df = pd.DataFrame(data)
    species_cols = df.columns[1:]
    relative_df = pd.DataFrame()
    relative_df["Week"] = df["Week"]
    row_sums = df[species_cols].sum(axis=1)
    for col in species_cols:
        relative_df[col] = df[col] / row_sums
    return relative_df

greenhouse_rel = calculate_relative_abundance(greenhouse)
animal_house_rel = calculate_relative_abundance(animal_house)

Shannon = {}

def calculate_shannon(data,label):
  print(label)
  shannon_df = pd.DataFrame()
  shannon_df["Week"] = data["Week"]
  species_cols = data.columns[1:]
  for index, row in data.iterrows():
    data_row = row[species_cols]
    shannon_df.loc[index, "Species Count"] = len(data_row[data_row > 0])
  for col in species_cols:
    shannon_df[f"{col} Diversity"] = -data[col] * np.log(data[col])

  New_Df = pd.DataFrame()
  New_Df["Week"] = shannon_df["Week"]
  New_Df["Species Count"] = shannon_df["Species Count"]
  New_Df["Shannon Diversity"] = shannon_df.filter(regex="Diversity$").sum(axis=1)
  New_Df["Equitability"] = shannon_df.filter(regex='Diversity$').sum(axis=1, skipna=True) / np.log(shannon_df['Species Count'])
  Shannon[label] = New_Df
  
  # print(f"Diversity: \n {shannon_df.filter(regex="Diversity$").sum(axis=1)}")
  # print(f"Species Count: \n {shannon_df['Species Count']}")
  # print(f"Equitability: \n {shannon_df.filter(regex='Diversity$').sum(axis=1, skipna=True) / np.log(shannon_df['Species Count'])}")
  return shannon_df

greenhouse_shannon = calculate_shannon(greenhouse_rel, "Greenhouse")
animal_house_shannon = calculate_shannon(animal_house_rel, "Animal House")

# Plot diversity and equitability
# Improved Shannon Diversity and Equitability Plot
def plot_diversity_equitability(dfs_tuple, labels):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs_tuple)))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>']

    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        df['Shannon Diversity (H)'] = Shannon[label]["Shannon Diversity"]
        df['Equitability (J)'] = Shannon[label]["Equitability"]
        ax1.plot(df['Week'], df['Shannon Diversity (H)'], color=colors[i], marker=markers[i % len(markers)],
                 linestyle='-', linewidth=2, label=f'{label} - Shannon Diversity (H)')

    ax1.set_xlabel('Weeks', fontsize=12)
    ax1.set_ylabel('Shannon Diversity (H)', fontsize=12)
    ax1.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    ax2 = ax1.twinx()
    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        ax2.plot(df['Week'], df['Equitability (J)'], color=colors[i], marker=markers[i % len(markers)],
                 linestyle='--', linewidth=2, label=f'{label} - Equitability (J)')

    ax2.set_ylabel('Equitability (J)', fontsize=12)

    # Consistent y-limits for both metrics
    min_y = min(min(df['Shannon Diversity (H)'].min(), df['Equitability (J)'].min()) for df in dfs_tuple)
    max_y = max(max(df['Shannon Diversity (H)'].max(), df['Equitability (J)'].max()) for df in dfs_tuple)
    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)

    # Combined legend for solid and dotted lines
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    custom_lines = [
        plt.Line2D([0], [0], color='black', linestyle='-', label="Shannon Diversity (H)"),
        plt.Line2D([0], [0], color='black', linestyle='--', label="Equitability (J)")
    ]
    ax1.legend(lines1 + lines2 + custom_lines, labels1 + labels2 + ["Shannon Diversity", "Equitability"], 
               loc='upper left', bbox_to_anchor=(1.05, 1))

    ax1.set_xticks(df['Week'])
    ax1.set_xticklabels(df['Week'], rotation=45, ha='right')

    fig.suptitle('Diversity and Equitability Over Time Across Multiple Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/diversity_equitability_over_time_improved.png")
    plt.close()

# Call the function

plot_diversity_equitability((greenhouse_shannon, animal_house_shannon), ["Greenhouse", "Animal House"])

# Plot Observed Individuals Over Time
def plot_observations(dfs_tuple, labels):
    num_plots = len(dfs_tuple)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots), sharex=True)

    for i, (df, label, ax) in enumerate(zip(dfs_tuple, labels, axes)):
        for col in df.columns[1:]:
            ax.plot(df['Week'], df[col], label=col, marker='o', linestyle='-')

        ax.set_ylabel('Observed Individuals', fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.legend(loc="upper left")
        ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    axes[-1].set_xlabel('Weeks', fontsize=12)
    fig.suptitle('Observed Individuals Over Time (Separate Y-Axes)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/observed_individuals_over_time_separate_y.png")
    plt.close()
plot_observations((greenhouse, animal_house), ["Greenhouse", "Animal House"])

# Plot Rank Abundance
def plot_rank_abundance(dfs_tuple, labels):
    num_plots = len(dfs_tuple)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots + 1, 7), sharey=True)
    all_species = set()
    for df in dfs_tuple:
        all_species.update(df.columns)
    all_species = sorted([species for species in all_species if species != "Week"])

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_species)))
    color_map = dict(zip(all_species, colors))

    for i, (df, ax, label) in enumerate(zip(dfs_tuple, axes, labels)):
        species_cols = [col for col in df.columns if col != "Week"]
        avg_rel_abundance = df[species_cols].mean().sort_values(ascending=False)
        ranks = range(1, len(avg_rel_abundance) + 1)
        bar_colors = [color_map[species] for species in avg_rel_abundance.index]

        ax.bar(ranks, avg_rel_abundance, color=bar_colors, edgecolor='k', alpha=0.8)
        ax.set_xlabel("Rank (Abundance)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Average Relative Abundance", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xticks(ranks)
        ax.set_xticklabels(ranks)
        ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    legend_labels = [species for species in all_species]
    legend_handles = [plt.Line2D([0], [0], color=color_map[species], marker='o', linestyle='', markersize=8) for species in all_species]
    fig.legend(legend_handles, legend_labels, title="Species", loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.suptitle("Average Rank Abundance Bar Plots (spider Data)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    plt.savefig(f"{output_dir}/rank_abundance_bar_plots_spiders.png")
    plt.close()

# Call the function for rank abundance plot
plot_rank_abundance((greenhouse_rel, animal_house_rel), ["Greenhouse", "Animal House"])


def plot_individual_diversity_equitability(df, label):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    df['Shannon Diversity (H)'] = Shannon[label]["Shannon Diversity"]
    df['Equitability (J)'] = Shannon[label]["Equitability"]

    # Calculate the range for shared y-axis scaling
    combined_min = min(df['Shannon Diversity (H)'].min(), df['Equitability (J)'].min())
    combined_max = max(df['Shannon Diversity (H)'].max(), df['Equitability (J)'].max())

    # Shannon Diversity
    ax1.plot(df['Week'], df['Shannon Diversity (H)'], marker='o', linestyle='-', color='blue', label="Shannon Diversity (H)")
    ax1.set_xlabel('Weeks', fontsize=12)
    ax1.set_ylabel('Shannon Diversity (H)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(combined_min, combined_max)
    ax1.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Equitability (on the second axis)
    ax2 = ax1.twinx()
    ax2.plot(df['Week'], df['Equitability (J)'], marker='s', linestyle='--', color='green', label="Equitability (J)")
    ax2.set_ylabel('Equitability (J)', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(combined_min, combined_max)

    # Titles and Legends
    fig.suptitle(f'Diversity and Equitability Over Time: {label}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/individual/diversity_equitability_{label.lower().replace(' ', '_')}.png")
    plt.close()


# Generate Individual Plots
plot_individual_diversity_equitability(Shannon["Greenhouse"], "Greenhouse")
plot_individual_diversity_equitability(Shannon["Animal House"], "Animal House")


def plot_individual_observations(df, label):
    fig, ax = plt.subplots(figsize=(8, 5))

    for col in df.columns[1:]:
        ax.plot(df['Week'], df[col], marker='o', linestyle='-', label=col)

    ax.set_xlabel('Weeks', fontsize=12)
    ax.set_ylabel('Observed Individuals', fontsize=12)
    ax.set_title(f'Observed Individuals Over Time: {label}', fontsize=14, fontweight='bold')
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual/observed_individuals_{label.lower().replace(' ', '_')}.png")
    plt.close()

# Generate Individual Observation Plots
plot_individual_observations(greenhouse, "Greenhouse")
plot_individual_observations(animal_house, "Animal House")


def plot_individual_rank_abundance(df, label):
    species_cols = [col for col in df.columns if col != "Week"]
    avg_rel_abundance = df[species_cols].mean().sort_values(ascending=False)
    ranks = range(1, len(avg_rel_abundance) + 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(avg_rel_abundance)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ranks, avg_rel_abundance, color=colors, edgecolor='k', alpha=0.8)
    ax.set_xlabel("Rank (Abundance)", fontsize=12)
    ax.set_ylabel("Average Relative Abundance", fontsize=12)
    ax.set_title(f'Rank Abundance: {label}', fontsize=14, fontweight='bold')
    ax.set_xticks(ranks)
    ax.set_xticklabels(ranks)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Species Legend
    legend_labels = avg_rel_abundance.index
    legend_handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='', markersize=8) for color in colors]
    ax.legend(legend_handles, legend_labels, title="Species", loc='upper right', bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual/rank_abundance_{label.lower().replace(' ', '_')}.png")
    plt.close()

# Generate Individual Rank Abundance Plots
plot_individual_rank_abundance(greenhouse_rel, "Greenhouse")
plot_individual_rank_abundance(animal_house_rel, "Animal House")


