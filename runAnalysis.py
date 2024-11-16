import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/individuals", exist_ok=True)

print(f"Reading {sys.argv[1]}...")

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

print(data)

(childrens_park, cricket_ground, shopping_complex) = (pd.DataFrame(data["Playground"]), pd.DataFrame(data["Cricket Ground"]), pd.DataFrame(data["Shopping Complex"]))

childrens_park.name = "Childrens Park"
cricket_ground.name = "Cricket Ground"
shopping_complex.name = "Shopping Complex"

class Population():
  def __init__(self,
               name,
               data: list[pd.DataFrame],
               scale = 1,
               separate_communities = False
  ):
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

  def create_sample_pop(self,
  ) -> dict:
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
    return

  def get_random_obs(self,
                     total_count,
  ) -> pd.DataFrame:
    if self.separate_communities:
      obs = []
      for community in self.sample_pop:
        total_individuals = community.values.sum()
        species_probs = (community.values / total_individuals)[0]
        # print(community.columns)
        # print(species_probs)
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
      # print(self.sample_pop.columns)
      # print(species_probs)
      selected_sample = np.random.choice(
          self.sample_pop.columns,
          size=total_count,
          p=species_probs
      )
      specii, counts = np.unique(selected_sample, return_counts=True)
      counts = dict(zip(specii, counts))
    return counts

total_population = Population(
    "Total Population",
     [childrens_park, cricket_ground, shopping_complex],
     separate_communities=True
    )

print(total_population.species)

seed = 42
np.random.seed(seed)
trials_per_point = 30
total_individuals_x_axes = np.arange(0, 1000)
num_species_y_axes = []
for total_individuals in tqdm(total_individuals_x_axes):
  num_species = np.zeros(len(total_population.data))
  for _ in range(trials_per_point):
    obs = total_population.get_random_obs(total_individuals)
    num_species += [len(x.values()) for x in obs]
  num_species /= trials_per_point
  num_species_y_axes.append(np.copy(num_species))

num_species_y_axes = np.array(num_species_y_axes)

# plot num_species vs total_individuals for each community as a line
fig, ax = plt.subplots(figsize=(10, 7))
for i, community in enumerate(total_population.data):
    ax.plot(total_individuals_x_axes, num_species_y_axes[:, i], label=community.name)
fig.suptitle("Number of Species vs Total Individuals", fontsize=14, fontweight='bold')
ax.set_xlabel("Total Individuals", fontsize=12)
ax.set_ylabel("Number of Species", fontsize=12)
ax.legend()
ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
plt.savefig(f"{output_dir}/num_species_vs_total_individuals.png")
plt.close()

def plot_observations(dfs_tuple, labels):
    num_plots = len(dfs_tuple)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)

    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        for col in df.columns[1:]:
            axes[i].plot(df['Week'], df[col], label=col)

        axes[i].set_xlabel('Weeks', fontsize=12)
        axes[i].set_ylabel('Observed Individuals', fontsize=12)
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    fig.suptitle('Observed Individuals Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/observed_individuals_over_time.png")
    plt.close()

plot_observations((childrens_park, cricket_ground, shopping_complex),["Childrens Park", "Cricket Ground", "Shopping Complex"])

def calculate_relative_abundance(data):
    df = pd.DataFrame(data)
    species_cols = df.columns[1:]
    relative_df = pd.DataFrame()
    relative_df["Week"] = df["Week"]
    row_sums = df[species_cols].sum(axis=1)
    for col in species_cols:
        relative_df[col] = df[col] / row_sums
    return relative_df

(childrens_park_rel, cricket_ground_rel, shopping_complex_rel) = (calculate_relative_abundance(childrens_park),calculate_relative_abundance(cricket_ground),calculate_relative_abundance(shopping_complex))

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

(childrens_park_shannon, cricket_ground_shannon, shopping_complex_shannon) = (calculate_shannon(childrens_park_rel,"Childrens Park"),calculate_shannon(cricket_ground_rel, "Cricket Ground"),calculate_shannon(shopping_complex_rel, "Shopping Complex"))

def plot_rank_abundance(dfs_tuple, labels):
    output_dir = "output_plots"
    num_plots = len(dfs_tuple)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)
    all_species = set()
    for df in dfs_tuple:
        all_species.update(df.columns)
    all_species = sorted([species for species in all_species if species != "Week"])

    # Use Seaborn's color_palette for vibrant colors
    colors = sns.color_palette("hls", len(all_species))
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
    fig.legend(legend_handles, legend_labels, title="Species", loc='upper right', bbox_to_anchor=(1, 1))

    plt.suptitle("Average Rank Abundance Bar Plots", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.99])
    plt.savefig(f"{output_dir}/rank_abundance_bar_plots.png")
    plt.close()


plot_rank_abundance((cricket_ground_rel, shopping_complex_rel, childrens_park_rel),labels=["Cricket Ground","Shopping Complex","Childrens Park"])


def plot_diversity_equitability(dfs_tuple, labels, output_dir=output_dir):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("hls", len(dfs_tuple))  # Use HLS color palette
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>']

    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        df['Shannon Diversity (H)'] = Shannon[label]["Shannon Diversity"]
        df['Equitability (J)'] = Shannon[label]["Equitability"]

        # Plot Shannon Diversity (H)
        ax1.plot(df['Week'], df['Shannon Diversity (H)'], color=colors[i], marker=markers[i % len(markers)], linestyle='-',
                 linewidth=2, label=f'{label} - Shannon Diversity (H)')

    ax1.set_xlabel('Weeks', fontsize=12)
    ax1.set_ylabel('Shannon Diversity (H)', fontsize=12)
    ax1.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    ax2 = ax1.twinx()
    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        # Plot Equitability (J)
        ax2.plot(df['Week'], df['Equitability (J)'], color=colors[i], marker=markers[i % len(markers)], linestyle='--',
                 linewidth=2, label=f'{label} - Equitability (J)')

    ax2.set_ylabel('Equitability (J)', fontsize=12)

    # Set y-axis limits
    min_y = min(min(df['Shannon Diversity (H)'].min(), df['Equitability (J)'].min()) for df in dfs_tuple)
    max_y = max(max(df['Shannon Diversity (H)'].max(), df['Equitability (J)'].max()) for df in dfs_tuple)
    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)

    # Create a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    custom_lines = [
        plt.Line2D([0], [0], color='black', linestyle='-', label="Shannon Diversity (H)"),
        plt.Line2D([0], [0], color='black', linestyle='--', label="Equitability (J)")
    ]
    ax1.legend(lines1 + lines2 + custom_lines, labels1 + labels2 + ["Shannon Diversity", "Equitability"], loc='upper left', bbox_to_anchor=(1.1, 1))
    ax1.set_xticks(df['Week'])
    ax1.set_xticklabels(df['Week'], rotation=45, ha='right')
    # Set the title and layout
    fig.suptitle('Diversity and Equitability Over Time Across Multiple Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plt.savefig(f"{output_dir}/diversity_equitability_over_time.png")
    plt.close()


# Define areas in square meters
areas = {
    "Childrens Park": 947.24,
    "Cricket Ground": 668.384,
    "Shopping Complex": 414.899
}

# Function to calculate population density
def calculate_population_density(df, area):
    density_df = pd.DataFrame()
    density_df["Week"] = df["Week"]
    density_df["Total Individuals"] = df.iloc[:, 1:].sum(axis=1)
    density_df["Population Density"] = density_df["Total Individuals"] / area
    return density_df

# Calculate population density for each location
childrens_park_density = calculate_population_density(childrens_park, areas["Childrens Park"])
cricket_ground_density = calculate_population_density(cricket_ground, areas["Cricket Ground"])
shopping_complex_density = calculate_population_density(shopping_complex, areas["Shopping Complex"])

# Function to plot population density
def plot_population_density(dfs_tuple, labels, output_dir=output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("hls", len(dfs_tuple))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>']

    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        ax.plot(df["Week"], df["Population Density"], color=colors[i], marker=markers[i % len(markers)], linestyle='-',
                linewidth=2, label=label)

    ax.set_xlabel('Weeks', fontsize=12)
    ax.set_ylabel('Population Density (Individuals/m²)', fontsize=12)
    ax.set_title('Population Density Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(df["Week"])
    ax.set_xticklabels(df["Week"], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/population_density_over_time.png")
    plt.close()

# Plot population density for all areas
plot_population_density(
    (childrens_park_density, cricket_ground_density, shopping_complex_density),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)

plot_diversity_equitability((cricket_ground_shannon,childrens_park_shannon,shopping_complex_shannon),["Cricket Ground","Childrens Park", "Shopping Complex"])

# Function to calculate population density per species
def calculate_species_density(df, area):
    density_df = pd.DataFrame()
    density_df["Week"] = df["Week"]
    for species in df.columns[1:]:
        density_df[f"{species} Density"] = df[species] / area
    return density_df

# Calculate species-wise density for each location
childrens_park_species_density = calculate_species_density(childrens_park, areas["Childrens Park"])
cricket_ground_species_density = calculate_species_density(cricket_ground, areas["Cricket Ground"])
shopping_complex_species_density = calculate_species_density(shopping_complex, areas["Shopping Complex"])

childrens_park_species_density.to_csv("ChildrensPark.csv")
cricket_ground_species_density.to_csv("CricketGround.csv")
shopping_complex_species_density.to_csv("Complex.csv")

# Function to plot species-wise density
def plot_species_density(dfs_tuple, labels, output_dir=output_dir):
    all_species = set()
    for df in dfs_tuple:
        all_species.update(col.replace(" Density", "") for col in df.columns if "Density" in col)

    all_species = sorted(all_species)
    num_species = len(all_species)
    colors = sns.color_palette("hls", num_species)

    fig, axes = plt.subplots(1, len(dfs_tuple), figsize=(6 * len(dfs_tuple), 6), sharey=True)
    for i, (df, label, ax) in enumerate(zip(dfs_tuple, labels, axes)):
        for species, color in zip(all_species, colors):
            species_density_col = f"{species} Density"
            if species_density_col in df.columns:
                ax.plot(df["Week"], df[species_density_col], label=species, color=color, linestyle='-', marker='o')
        
        ax.set_xlabel("Weeks", fontsize=12)
        if i == 0:
            ax.set_ylabel("Density (Individuals/m²)", fontsize=12)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle("Population Density Per Species Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(f"{output_dir}/species_population_density_over_time.png")
    plt.close()

# Plot species-wise density for all locations
plot_species_density(
    (childrens_park_species_density, cricket_ground_species_density, shopping_complex_species_density),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)


def plot_individual_observations(dfs_tuple, labels, output_dir=output_dir):
    for df, label in zip(dfs_tuple, labels):
        fig, ax = plt.subplots(figsize=(8, 6))
        for col in df.columns[1:]:
            ax.plot(df['Week'], df[col], label=col)

        ax.set_xlabel('Weeks', fontsize=12)
        ax.set_ylabel('Observed Individuals', fontsize=12)
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.legend(title="Species")
        ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/individuals/observed_individuals_{label.replace(' ', '_').lower()}.png")
        plt.legend()
        plt.close()


def plot_individual_rank_abundance(dfs_tuple, labels, output_dir=output_dir):
    all_species = set()
    for df in dfs_tuple:
        all_species.update(df.columns)
    all_species = sorted([species for species in all_species if species != "Week"])

    colors = sns.color_palette("hls", len(all_species))
    color_map = dict(zip(all_species, colors))

    for df, label in zip(dfs_tuple, labels):
        species_cols = [col for col in df.columns if col != "Week"]
        avg_rel_abundance = df[species_cols].mean().sort_values(ascending=False)
        ranks = range(1, len(avg_rel_abundance) + 1)
        bar_colors = [color_map[species] for species in avg_rel_abundance.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(ranks, avg_rel_abundance, color=bar_colors, edgecolor='k', alpha=0.8)
        ax.set_xlabel("Rank (Abundance)", fontsize=12)
        ax.set_ylabel("Average Relative Abundance", fontsize=12)
        ax.set_title(f"Rank Abundance - {label}", fontsize=14, fontweight='bold')
        ax.set_xticks(ranks)
        ax.set_xticklabels(ranks)
        ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/individuals/rank_abundance_{label.replace(' ', '_').lower()}.png")
        plt.legend()
        plt.close()


def plot_individual_diversity_equitability(dfs_tuple, labels, output_dir=output_dir):
    colors = sns.color_palette("hls", len(dfs_tuple))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'H', 'v', '<', '>']

    for i, (df, label) in enumerate(zip(dfs_tuple, labels)):
        fig, ax1 = plt.subplots(figsize=(8, 6))
        df['Shannon Diversity (H)'] = Shannon[label]["Shannon Diversity"]
        df['Equitability (J)'] = Shannon[label]["Equitability"]

        # Plot Shannon Diversity
        ax1.plot(df['Week'], df['Shannon Diversity (H)'], color=colors[i], marker=markers[i % len(markers)], linestyle='-',
                 linewidth=2, label='Shannon Diversity (H)')
        ax1.set_xlabel('Weeks', fontsize=12)
        ax1.set_ylabel('Shannon Diversity (H)', fontsize=12)
        ax1.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

        ax2 = ax1.twinx()
        # Plot Equitability
        ax2.plot(df['Week'], df['Equitability (J)'], color=colors[i], marker=markers[i % len(markers)], linestyle='--',
                 linewidth=2, label='Equitability (J)')
        ax2.set_ylabel('Equitability (J)', fontsize=12)

        fig.suptitle(f"Diversity and Equitability - {label}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/individuals/diversity_equitability_{label.replace(' ', '_').lower()}.png")
        plt.legend()
        plt.close()


def plot_individual_species_density(dfs_tuple, labels, output_dir=output_dir):
    all_species = set()
    for df in dfs_tuple:
        all_species.update(col.replace(" Density", "") for col in df.columns if "Density" in col)

    all_species = sorted(all_species)
    colors = sns.color_palette("hls", len(all_species))

    for df, label in zip(dfs_tuple, labels):
        fig, ax = plt.subplots(figsize=(8, 6))
        for species, color in zip(all_species, colors):
            species_density_col = f"{species} Density"
            if species_density_col in df.columns:
                ax.plot(df["Week"], df[species_density_col], label=species, color=color, linestyle='-', marker='o')

        ax.set_xlabel("Weeks", fontsize=12)
        ax.set_ylabel("Density (Individuals/m²)", fontsize=12)
        ax.set_title(f"Species Density - {label}", fontsize=14, fontweight="bold")
        ax.grid(visible=True, color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="Species")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/individuals/species_density_{label.replace(' ', '_').lower()}.png")
        plt.legend(loc="best")
        plt.close()

plot_individual_observations(
    (childrens_park, cricket_ground, shopping_complex),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)

plot_individual_rank_abundance(
    (childrens_park_rel, cricket_ground_rel, shopping_complex_rel),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)

plot_individual_diversity_equitability(
    (childrens_park_shannon, cricket_ground_shannon, shopping_complex_shannon),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)

plot_individual_species_density(
    (childrens_park_species_density, cricket_ground_species_density, shopping_complex_species_density),
    ["Childrens Park", "Cricket Ground", "Shopping Complex"]
)


def plot_combined_diversity_equitability(weeks, labels):

    # Colors and markers for different lines
    colors = plt.cm.tab10.colors  # Predefined color palette
    markers = ['o', 's', 'D', '^', 'v', '<', '>', '*', 'h', 'p']  # Different markers

    # Combined Diversity Plot
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(labels):
        plt.plot(
            weeks,
            Shannon[label]["Shannon Diversity"],
            marker=markers[idx % len(markers)],
            linestyle='-',
            color=colors[idx % len(colors)],
            label=label
        )
    plt.title(' Shannon Diversity (H) Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Shannon Diversity (H)', fontsize=12)
    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_shannon_diversity.png")
    plt.close()

    # Combined Equitability Plot
    plt.figure(figsize=(10, 6))
    for idx, label in enumerate(labels):
        plt.plot(
            weeks,
            Shannon[label]["Equitability"],
            marker=markers[idx % len(markers)],
            linestyle='-',
            color=colors[idx % len(colors)],
            label=label
        )
    plt.title('Equitability (J) Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Equitability (J)', fontsize=12)
    plt.legend()
    plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_equitability.png")
    plt.close()


plot_combined_diversity_equitability(weeks=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],labels=["Childrens Park", "Cricket Ground", "Shopping Complex"])
