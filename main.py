"""Minimal agent based economic simulation.

This script models a population of simple agents that harvest resources,
trade with one another and consume goods to gain rewards.  The environment is
designed purely for experimentation and therefore relies heavily on PyTorch for
automatic differentiation and optimisation of the agents' neural networks.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.optimize import linprog
import matplotlib.colors as mcolors  # make sure this is imported at the top
from random import randint

plt.switch_backend('agg')  # use non‑interactive backend for image generation

# CUDA is optional; fall back to CPU if unavailable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Global state trackers -------------------------------------------------
# History of total resources in the environment for plotting
resource_history: list = []

# Saved broadcasts from each timestep for later analysis
broadcast_history: list = []

# Numerical cutoff for treating tiny values as zero
small_value_threshold = 1e-5

#use_softmax_allocation = True  #  Toggle this for testing

# --- Simulation configuration ----------------------------------------------

# Number of distinct resources in the world
input_dim = 2 

# Human readable names for each resource
resource_names = ["Berries", "Meat"]

# Length of the vector each agent broadcasts to others
broadcast_dim = 6

# Number of available jobs/tasks in the environment
num_jobs = 2 

# Size of the agent population
num_agents = 30

# Reward given for resting instead of exerting effort
alpha = 0.005

# Parameters for projected gradient descent used during trading
pgd_steps = 100
pgd_lr = 0.01

# Exponential penalty on maximum effort as agents age
age_effort_penalty = 0.15

# History of inequality measure over time
gini_history: list = []

# Maximum lifespan of an agent (in steps)
max_age = 60

# Global rate at which unused resources decay each step
decay_rate = 0.1

#For an agent to survive to the next step, the inner product of their consumption and min_vector must be greater than min_const
min_vector = torch.tensor([1.0,1.0], device=device)
min_const = 0.1

#Cap on how much of each resource can be produced per step
resource_cap = torch.tensor([float("Inf"),float("Inf"),float("Inf")],device=device)

max_steps=1000
plot_freq = 1 #frequency at which a plot is saved for GIF generation
num_gif = 1
def init_jobs():
    """Define the available production jobs in the environment."""
    """
    return [
        {'name':'Gathering Berries','input': torch.tensor([0.0, 0.0,0.0], device=device),
         'output': torch.tensor([1.0,0.0,0.0], device=device),
         'phase':0.0,
         'period':float("Inf"),
         'min':0.0},
        {'name':'Trapping Game','input': torch.tensor([0.01, 0.0,0.0], device=device),
         'output': torch.tensor([0.0,1.5,1.0], device=device),
         'phase':1.57,
         'period':float("Inf"),
         'min':0.0},
        {'name':'Fishing','input':torch.tensor([0.0,0.01,0.0],device=device),
         'output':torch.tensor([0.0,2.0,0.0], device=device),
         'phase':0.0,
         'period':float("Inf"),
         'min':0.0}
        #{'name':'Hunting Deer','input': torch.tensor([0.1,0.1,0.0], device=device), 'output': torch.tensor([0.0,0.0,1.0], device=device)},
    ]
    """
    return [
            {'name':'Gathering',
             'input':torch.tensor([0.0,0.0],device=device),
             'output':torch.tensor([1.0,0.0],device=device),
             'phase':1.57,
             'period':100,
             'min':0.0},
            {'name':'Hunting',
             'input':torch.tensor([0.0,0.0],device=device),
             'output':torch.tensor([0.0,1.0],device=device),
             'phase':0.0,
             'period':100,
             'min':0.0},
            ]
#Each neural network of an agent is trained separately with its own optimizer
def make_agent_optimizers(agent):
    """Create optimizers for all sub-networks of a single agent."""
    return [
        torch.optim.Adam(agent.broadcast_net.parameters(), lr=0.05),
        torch.optim.Adam(agent.social_filter_net.parameters(), lr=0.05),
        torch.optim.Adam(agent.production_net.parameters(), lr=0.05),
        torch.optim.Adam(agent.trading_net.parameters(), lr=0.05),
        torch.optim.Adam(agent.consumption_net.parameters(), lr=0.05)
    ]



#Prunes underconsuming and old agents, replacing them with new agents that inherit traits from their parents
def prune_underconsuming_agents(agents, min_vector, mutation_strength=0.1):
    """Replace agents that consume too little or grow too old with offspring."""
    consumed_tensor = torch.stack([
        agent.cached_outputs.get("last_consumed", torch.zeros_like(min_vector)).squeeze(0) for agent in agents
    ])

    age_tensor = torch.tensor([agent.age for agent in agents], device=device)
    dot_products = torch.sum(consumed_tensor * min_vector, dim=1)
    keep_mask = (dot_products >= min_const) & (age_tensor < max_age)
    surviving_agents = [agent for i, agent in enumerate(agents) if keep_mask[i]]
    surviving_optimizers = [make_agent_optimizers(agent) for agent in surviving_agents]

    num_replace = len(agents) - len(surviving_agents)
    if num_replace == 0:
        return surviving_agents, surviving_optimizers

    if len(surviving_agents) >= 2:
        scores = torch.tensor([agent.accum_reward for agent in surviving_agents])
        probs = scores / scores.sum()
        top_agents = surviving_agents
    else:
        probs = None
        top_agents = []

    # Helper used when creating offspring from two surviving parents
    def breed(parent1, parent2):
        child = Agent().to(device)
        with torch.no_grad():
            child.resources = (
                0.3 * parent1.resources.detach().clone() +
                0.3 * parent2.resources.detach().clone()
            ).clone().to(device).requires_grad_(True)

            child.reward_weights = torch.softmax((parent1.reward_weights+parent2.reward_weights + mutation_strength*torch.rand_like(child.reward_weights)),dim=1)

            child.parent1 = parent1
            child.parent2 = parent2
            parent1.resources = (parent1.resources.detach() * 0.7).clone().requires_grad_(True)
            parent2.resources = (parent2.resources.detach() * 0.7).clone().requires_grad_(True)

            for c, a, b in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                if c.shape != a.shape or c.shape != b.shape:
                    continue
                alpha = torch.rand_like(c)
                blended = alpha * a.data + (1 - alpha) * b.data
                mutated = blended + torch.randn_like(blended) * mutation_strength
                c.data.copy_(mutated)

        return child, make_agent_optimizers(child)

    new_agents, new_optimizers = [], []
    for i in range(num_replace):  #OPTIMIZE THIS!!!!!!!!!!!!!!!!!!!!!!
        if len(top_agents) >= 2:
            indices = np.random.choice(len(top_agents), size=(num_replace, 2), p=probs.cpu().numpy(), replace=True)
            p1 = top_agents[indices[i, 0]]
            p2 = top_agents[indices[i, 1]]
            offspring, offspring_opt = breed(p1, p2)
        else:
            offspring = Agent().to(device)
            with torch.no_grad():
                for c in offspring.parameters():
                    mutated = c.data + alpha * torch.rand_like(c)
                    c.data.copy_(mutated)
            offspring_opt = make_agent_optimizers(offspring)
        new_agents.append(offspring)
        new_optimizers.append(offspring_opt)

    # Distribute dead agents' resources to their children
    dead_agents = [agent for i, agent in enumerate(agents) if not keep_mask[i]]
    child_map = {agent: [] for agent in dead_agents}
    for child in new_agents:
        if child.parent1 and child.parent1 in child_map:
            child_map[child.parent1].append(child)
        if child.parent2 and child.parent2 in child_map:
            child_map[child.parent2].append(child)
    for parent, children in child_map.items():
        if not children:
            continue
        share = (parent.resources.detach() / len(children)).clone()
        for child in children:
            child.resources = (child.resources.detach() + share).clone().requires_grad_(True)
            if child.parent1 is parent:
                child.parent1 = None
            if child.parent2 is parent:
                child.parent2 = None
        parent.resources = torch.zeros_like(parent.resources).requires_grad_(True)

    return surviving_agents + new_agents, surviving_optimizers + new_optimizers



    # === Visualization util ===

def add_broadcast_pca_colored_subplot(broadcasts, agents, fig, axes, position, color_by="age"):
    """
    Adds a PCA subplot of broadcasts, color-coded by a given agent attribute,
    and sized by total resource pool.

    Parameters:
        broadcasts (torch.Tensor): (num_agents, broadcast_dim)
        agents (list): list of Agent objects
        fig (matplotlib.Figure)
        axes (list): flattened list of axes
        position (int): where to place the new subplot
        color_by (str): one of ['age', 'effort', 'consumption', 'production', 'rgb']
    """
    from sklearn.decomposition import PCA

    def normalize(x):
        x = np.asarray(x)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    # Convert to numpy
    if hasattr(broadcasts, "detach"):
        broadcasts = broadcasts.detach().cpu().numpy()

    # PCA projection
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(broadcasts)

    # Size of points based on resource pool
    sizes = np.array([agent.resources.sum().item() for agent in agents])
    sizes = normalize(sizes) * 200 + 10  # scale for visibility

    # Choose coloring
    if color_by == "rgb":
        R = normalize([agent.age for agent in agents])
        G = normalize([
            agent.cached_outputs.get("last_consumed", torch.zeros_like(agent.resources)).sum().item()
            for agent in agents
        ])
        B = normalize([agent.last_effort.item() for agent in agents])
        colors = np.stack([R, G, B], axis=1)
        label = "RGB: Age, Consumption, Effort"
        cbar = None  # skip colorbar for RGB mode
    else:
        if color_by == "age":
            values = np.array([agent.age for agent in agents])
            cmap = "plasma"
            label = "Agent Age"
            norm = mcolors.Normalize(vmin=0, vmax=max_age)
        elif color_by == "effort":
            values = np.array([agent.last_effort.item() for agent in agents])
            cmap = "plasma"
            label = "Effort"
            norm = mcolors.Normalize(vmin=0, vmax=1)
        elif color_by == "consumption":
            values = np.array([
                agent.cached_outputs.get("last_consumed", torch.zeros_like(agent.resources)).sum().item()
                for agent in agents
            ])
            cmap = "plasma"
            label = "Total Consumption"
            norm = mcolors.Normalize(vmin=0, vmax=np.percentile(values, 95))
        elif color_by == "production":
            values = np.array([
                agent.cached_outputs.get("last_produced", torch.zeros_like(agent.resources)).sum().item()
                for agent in agents
            ])
            cmap = "plasma"
            label = "Total Production"
            norm = mcolors.Normalize(vmin=0, vmax=np.percentile(values, 95))
        else:
            raise ValueError(f"Unsupported color_by: {color_by}")

        colors = values
        cbar = True

    # Plotting
    ax = axes[position]
    scatter = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=colors, s=sizes,
                         cmap=cmap if color_by != "rgb" else None,
                         norm=norm if color_by != "rgb" else None,
                         alpha=0.8)

    ax.set_title(f"Broadcasts (Color: {label})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    ax.grid(True)

    if cbar:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(label)

    return fig, axes
def add_broadcast_pca_colored_by_job(broadcasts, agents, chosen_jobs, fig, axes, position):
    """
    Adds a PCA plot of broadcasts colored by job ID (int), with padded -1 entries shown in gray.

    Parameters:
        broadcasts (Tensor): shape (N, D)
        agents (List): list of length N
        chosen_jobs (List or array): list of length N (or shorter/longer)
        fig, axes, position: matplotlib layout
        num_jobs (int): total number of defined jobs (excluding padded -1)
    """
    from sklearn.decomposition import PCA
    import matplotlib.colors as mcolors

    if hasattr(broadcasts, "detach"):
        broadcasts = broadcasts.detach().cpu().numpy()

    N = len(agents)
    chosen_jobs = np.array(chosen_jobs)

    # Fix length mismatch
    if len(chosen_jobs) != N:
        #print(f"[WARN] Job list has length {len(chosen_jobs)}, expected {N}")
        if len(chosen_jobs) > N:
            chosen_jobs = chosen_jobs[:N]
        else:
            chosen_jobs = np.pad(chosen_jobs, (0, N - len(chosen_jobs)), constant_values=-1)

    # Prepare PCA
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(broadcasts)

    # Agent sizes by total resources
    sizes = np.array([agent.resources.sum().item() for agent in agents])
    sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8)
    sizes = sizes * 200 + 10


    # Custom color map: tab10 + gray for -1
    base_colors = plt.get_cmap("tab10").colors
    cmap_colors = list(base_colors[:num_jobs]) + [(0.6, 0.6, 0.6)]  # gray at end
    cmap = mcolors.ListedColormap(cmap_colors)

    # Remap jobs: -1 → num_jobs (the last color, gray)
    color_indices = np.where(chosen_jobs == -1, num_jobs, chosen_jobs)

    # Plot
    ax = axes[position]
    scatter = ax.scatter(pca_proj[:, 0], pca_proj[:, 1],
                         c=color_indices, cmap=cmap, s=sizes, alpha=0.85)

    ax.set_title("Broadcasts Colored by Job")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)

    # Add colorbar
    ticks = list(range(num_jobs)) + [num_jobs]
    tick_labels = [str(i) for i in range(num_jobs)] + ["None"]
    cbar = fig.colorbar(scatter, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("Job ID")

    return fig, axes

def add_social_filter_heatmap_subplot(agents, broadcasts, fig, axes, position):
    """Add heatmap showing how each agent weights others via the social filter net."""
    num_agents = len(agents)

    if hasattr(broadcasts, "detach"):
        broadcasts = broadcasts.detach()

    weights = torch.zeros(num_agents, num_agents, device=broadcasts.device)

    for i, agent in enumerate(agents):
        other_indices = [j for j in range(num_agents) if j != i]
        if not other_indices:
            continue

        others_broadcasts = broadcasts[other_indices]
        ids_expanded = torch.stack([agents[j].id_vector.squeeze(0) for j in other_indices])
        own_broadcast_exp = broadcasts[i].expand_as(others_broadcasts)
        own_resources_exp = agent.resources.expand(len(other_indices), -1)
        other_resources = torch.stack([agents[j].resources for j in other_indices]).squeeze(1)

        pairwise = torch.cat([
            own_broadcast_exp, others_broadcasts,
            ids_expanded, other_resources, own_resources_exp
        ], dim=-1)

        scores = agent.social_filter_net(pairwise).squeeze(-1)
        probs = F.softmax(scores, dim=0)
        weights[i, other_indices] = probs.detach()

    matrix = weights.cpu().numpy()

    ax = axes[position]
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Social Filter Weights")
    ax.set_xlabel("Target Agent")
    ax.set_ylabel("Source Agent")
    fig.colorbar(im, ax=ax)

    return fig, axes



def add_effort_vs_consumption_subplot(agents, fig, axes, position):
    """Scatter plot comparing effort with total consumption for each agent."""
    efforts = [
        agent.last_effort.detach().cpu().item() if hasattr(agent, 'last_effort') else 0.0
        for agent in agents
    ]
    consumptions = [
        agent.cached_outputs["last_consumed"].sum().detach().cpu().item()
        if "last_consumed" in agent.cached_outputs else 0.0
        for agent in agents
    ]
    ages = [agent.age if hasattr(agent, 'age') else 0 for agent in agents]

    norm = mcolors.Normalize(vmin=0, vmax=max_age)

    ax = axes[position]
    scatter = ax.scatter(efforts, consumptions, c=ages, cmap='viridis', norm=norm, alpha=0.8)
    ax.set_title("Effort vs Total Consumption (by age)")
    ax.set_xlabel("Effort")
    ax.set_ylabel("Total Consumption")
    ax.grid(True)
    ax.set_xlim(-0.1,1)

    # Add colorbar with fixed scale
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Agent Age")
    cbar.set_ticks([0, max_age // 2, max_age])

    return fig, axes


def gini(array):
    """Compute Gini coefficient of a 1D torch tensor."""
    array = array.flatten()
    if array.numel() == 0:
        return 0.0
    sorted_array, _ = torch.sort(array)
    n = array.shape[0]
    index = torch.arange(1, n + 1, dtype=torch.float32, device=array.device)
    numerator = torch.sum((2 * index - n - 1) * sorted_array)
    denominator = n * torch.sum(sorted_array) + 1e-3  # avoid division by zero
    return (numerator / denominator).item()
def add_resource_totals_subplot(resource_history, step, fig, axes, position):
    """
    Adds a subplot showing the total amount of each resource in the society over time.

    Parameters:
        resource_history (list of list of floats): Each entry is a list of total resources at a time step
        step (int): Current step
        fig (matplotlib.Figure): Existing figure
        axes (list): Flattened list of axes
        position (int): Index in axes to place the new subplot
    """
    if len(resource_history)>0:
        steps = np.arange(len(resource_history))
        totals = np.array(resource_history)  # shape: (num_steps, num_resources)

        ax = axes[position]
        num_resources = totals.shape[1]
        
        for r in range(num_resources):
            ax.plot(steps, totals[:, r], label=resource_names[r])

        ax.set_xlabel('Step')
        ax.set_ylabel('Total Quantity')
        ax.set_title("Total Resources in Society")
        ax.set_xlim(max(0, step - 100), step)
        ax.legend()
        ax.grid(True)

    return fig, axes
def add_broadcast_eigenvalue_subplot(broadcasts, step, fig, axes, position):
    """
    Adds a subplot showing the magnitudes of eigenvalues of the broadcast matrix.

    Parameters:
        broadcasts (torch.Tensor): Tensor of shape (num_agents, broadcast_dim)
        step (int): Current simulation step
        fig (matplotlib.Figure): Existing figure
        axes (list): Flattened list of axes
        position (int): Index in axes to place the new subplot
    """
    import numpy as np

    # Detach and convert to NumPy
    matrix = broadcasts.detach().cpu().numpy()

    # Compute eigenvalues of the covariance matrix (safer for rectangular matrices)
    cov = np.cov(matrix, rowvar=False)
    eigvals = np.linalg.eigvals(cov)
    eig_magnitudes = np.abs(eigvals)
    eig_magnitudes.sort()
    ax = axes[position]
    ax.plot(np.arange(len(eig_magnitudes)), eig_magnitudes, marker='o', linestyle='-')
    ax.set_title(f'Eigenvalue Magnitudes of Broadcasts)')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Magnitude')
    ax.grid(True)

    return fig, axes

def plot_trade_with_supply_demand(before, after, step, agents, broadcasts, job_names=None, chosen_jobs=None):
    """Visualize trading results along with several diagnostic subplots."""
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    before = before.cpu().numpy() if hasattr(before, 'cpu') else np.array(before)
    after = after.cpu().numpy() if hasattr(after, 'cpu') else np.array(after)
    delta = after - before
    num_resources = before.shape[1]

    # Determine number of resource subplot pairs
    pairs = [(i, j) for i in range(num_resources) for j in range(i + 1, num_resources)]
    num_pairs = len(pairs)

    # Total subplots: resource pairs + age histogram + supply/demand + job bar chart
    total_plots = num_pairs + 14
    ncols = 4
    nrows = (total_plots + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()
    # Add resource total subplot at the next available index
    if step > 1:
        fig, axes = add_resource_totals_subplot(resource_history, step, fig, axes, position=total_plots - 2)
        fig, axes = add_broadcast_eigenvalue_subplot(broadcasts, step, fig, axes, position=total_plots - 1)
        fig, axes = add_effort_vs_consumption_subplot(agents, fig, axes, position=total_plots - 3)
        fig, axes = add_broadcast_pca_colored_subplot(broadcasts, agents, fig, axes, position=total_plots - 4, color_by="age")
        fig, axes = add_broadcast_pca_colored_subplot(broadcasts, agents, fig, axes, position=total_plots - 5, color_by="effort")
        fig, axes = add_broadcast_pca_colored_subplot(broadcasts, agents, fig, axes, position=total_plots - 6, color_by="consumption")
        fig, axes = add_broadcast_pca_colored_subplot(broadcasts, agents, fig, axes, position=total_plots - 7, color_by="production")
        fig, axes = add_broadcast_pca_colored_by_job(broadcasts, agents, chosen_jobs, fig, axes, position=total_plots - 8)
        fig, axes = add_social_filter_heatmap_subplot(agents, broadcasts, fig, axes, position=total_plots - 9)



    # Resource scatter subplots
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(before[:, i], before[:, j], c='red', label='Before Trade')
        ax.scatter(after[:, i], after[:, j], c='green', label='After Trade')
        #ax.set_xscale("symlog")
        #ax.set_yscale("symlog")
        ax.set_xlim(-0.1, max(0.5, np.max(np.concatenate([before[:, i], after[:, i]]))))
        ax.set_ylim(-0.1, max(0.5, np.max(np.concatenate([before[:, j], after[:, j]]))))

        for k in range(before.shape[0]):
            ax.plot([before[k, i], after[k, i]], [before[k, j], after[k, j]], 'gray', linewidth=0.65, alpha=0.6)
        ax.set_xlabel(resource_names[i])
        ax.set_ylabel(resource_names[j])
        ax.set_title(f'{resource_names[i]} vs {resource_names[j]}')
        ax.legend()

    # Age histogram subplot
    ax = axes[num_pairs]
    ages = [agent.age for agent in agents]
    ax.hist(ages, bins=range(0, max_age), color='blue', alpha=0.7, edgecolor='black')
    ax.set_title("Agent Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_ylim(0, len(agents))
    ax.set_xlim(0, max_age)

    # Supply/Demand subplot
    ax = axes[num_pairs + 1]
    init_list = []
    dem_list = []
    for r in range(num_resources):
        initial = before[:, r]
        demand = delta[:, r]
        init_list.append(initial)
        dem_list.append(demand)
        ax.scatter(initial, demand, alpha=0.6, label=f'{resource_names[r]}')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Supply/Demand Across Resources")
    ax.set_xlabel("Initial Holdings")
    ax.set_ylabel("Net Demand (After - Before)")

    #ax.set_xlim(-0.1, max(0.5,np.max(init_list)))
    #ax.set_ylim(min(-0.5,-np.max(np.abs(dem_list)) ), max(0.5,np.max(np.abs(dem_list)) ))
    ax.legend()
    ax.grid(True)

    # Job selection bar chart
    ax = axes[num_pairs + 2]
    if chosen_jobs is not None and job_names is not None:
        job_counts = np.bincount(chosen_jobs, minlength=len(job_names))
        ax.barh(range(len(job_names)), job_counts, color='purple', alpha=0.7)
        ax.set_yticks(range(len(job_names)))
        ax.set_yticklabels(job_names)
        ax.set_xlabel("Agent Count")
        ax.set_title("Job Choices This Step")
        ax.set_xlim(0,num_agents)
    else:
        ax.set_visible(False)

    # Remove unused subplots
    for k in range(total_plots, len(axes)):
        fig.delaxes(axes[k])

    plt.suptitle(f'Resource Distributions, Age, Supply/Demand, and Job Choices at Step {step}', fontsize=16)
    os.makedirs("trade_scatter", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"trade_scatter/step_{step:04d}.png")
    plt.close()


def generate_trade_gif():
    """Combine saved trade scatter plots into an animated GIF."""
    image_dir = "trade_scatter"
    gif_path = os.path.join(image_dir, "trade_evolution.gif")
    images = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png"):
            path = os.path.join(image_dir, fname)
            images.append(imageio.imread(path))
    if images:
        imageio.mimsave(gif_path, images, duration=0.2)
        print(f"Saved animated gif to {gif_path}")
    else:
        print("No images found for GIF generation.")

def seasonal_job_output(job, step, job_idx):
    """Return seasonally adjusted job output (currently unused)."""
    step_tensor = torch.tensor(0.01 * step + job_idx, device=device)
    scale = 1.0 + 0.5 * torch.sin(step_tensor)
    return job['output'] #* scale


class FlexibleMLP(nn.Module):
    """Simple configurable multi-layer perceptron used for all agent networks."""

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden_layers=2, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_hidden_layers > 0:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_layer = nn.Linear(input_dim, output_dim)
        self.activation_fn = F.relu if activation == 'relu' else torch.sigmoid if activation == 'sig' else torch.tanh

    def forward(self, x):
        """Run a forward pass through all layers."""
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.output_layer(x)

class Agent(nn.Module):
    """Encapsulates all neural networks and state for a single agent."""
    def __init__(self):
        super().__init__()
        self.age = 0
        self.max_age = max_age
        self.resources = torch.zeros(1, input_dim, device=device)
        self.id_vector = torch.randn(1, broadcast_dim, device=device)
        self.broadcast_net = FlexibleMLP(input_dim, broadcast_dim, 32, 2, 'tanh')  #neural net for agent broadcasting
        self.social_filter_net = FlexibleMLP(broadcast_dim * 3 + input_dim * 2, 1, 64, 2) #for filtering all other agents broadcasts
        self.production_net = FlexibleMLP(input_dim + broadcast_dim, num_jobs + 1, 32, 2) #for determining the agents job and effort
        self.trading_net = FlexibleMLP(input_dim + broadcast_dim, input_dim, 32, 1,'sig') #for determining the agents preferences during trading
        self.consumption_net = FlexibleMLP(input_dim + broadcast_dim, input_dim, 32, 2) #for determining what fraction of the agents resources it should consume
        self.cached_outputs = {}
        self.last_effort = torch.tensor(0.0, device=device)
        self.reward_baseline = 0.0
        self._init_weights()
        self.parent1 = None
        self.parent2 = None
        self.accum_reward = 0.0
        self.reward_weights = torch.softmax(torch.rand(1,input_dim),dim=1) #Reward weights are genetic and determine how the agent is rewarded during the consumption stage
    def _init_weights(self):
        """Initialise all network weights using Xavier uniform distribution."""
        for net in [self.broadcast_net, self.social_filter_net, self.production_net, self.trading_net, self.consumption_net]:
            for param in net.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)  # or add small noise here





def compute_social_signals(agents, broadcasts):
    """Aggregate other agents' broadcasts into a single signal per agent."""
    signals = []
    for i, agent in enumerate(agents):
        others_idx = [j for j in range(num_agents) if j != i]
        others_broadcasts = broadcasts[others_idx]
        ids_expanded = torch.stack([agents[j].id_vector.squeeze(0) for j in others_idx])
        own_broadcast_exp = broadcasts[i].expand_as(others_broadcasts)
        own_resources_exp = agent.resources.expand(len(others_idx), -1)
        other_resources = torch.stack([agents[j].resources for j in others_idx]).squeeze(1)
        pairwise = torch.cat([
            own_broadcast_exp, others_broadcasts,
            ids_expanded, other_resources, own_resources_exp
        ], dim=-1)
        scores = agent.social_filter_net(pairwise).view(1, -1)
        weights = F.softmax(scores, dim=-1)
        signal = torch.sum(others_broadcasts * weights.squeeze(0).unsqueeze(-1), dim=0, keepdim=True)
        signals.append(signal)
    return signals
def project_columns_onto_simplex(X, totals):
    """
    Projects each column of X (N x R) onto the simplex {x | x >= 0, sum(x) = totals[r]}
    """
    X_proj = X.clone()
    N, R = X.shape
    for r in range(R):
        if totals[r] <= 1e-8:  # safeguard for total = 0
            X_proj[:, r] = 0.0
            continue

        col = X[:, r]
        sorted_vals, _ = torch.sort(col, descending=True)
        cssv = torch.cumsum(sorted_vals, dim=0) - totals[r]
        range_vals = torch.arange(1, N + 1, device=X.device, dtype=col.dtype)
        condition = sorted_vals * range_vals > cssv
        indices = torch.nonzero(condition, as_tuple=False)

        if indices.numel() == 0:
            theta = 0.0
        else:
            rho = indices[-1].item()
            theta = cssv[rho] / (rho + 1)

        X_proj[:, r] = torch.clamp(col - theta, min=0.0)


    return X_proj


# Simplex projection function (differentiable)
def project_to_simplex(v, eps=1e-6):
    """Project a vector onto the probability simplex."""
    if v.dim() == 1:
        v = v.unsqueeze(0)

    v_sorted, _ = torch.sort(v, descending=True, dim=1)
    cssv = torch.cumsum(v_sorted, dim=1)
    rho = torch.arange(1, v.shape[1] + 1, device=v.device).float()
    cond = v_sorted * rho > (cssv - 1)
    k = cond.sum(dim=1, keepdim=True)
    tau = (cssv.gather(1, k - 1) - 1) / k
    w = torch.clamp(v - tau, min=0)
    return w.squeeze(0) if w.shape[0] == 1 else w

# PGD trading using detached computation on the global device
def run_trading_pgd(agents, social_signals):
    """Optimize resource distribution between agents via projected gradient descent."""
    num_agents = len(agents)
    input_dim = agents[0].resources.shape[1]

    init_resources = torch.stack([agent.resources.detach().to(device).squeeze(0) for agent in agents])
    current_resources = init_resources.clone().detach().requires_grad_(True)

    initial_utilities = []
    trading_weights = []

    for i, agent in enumerate(agents):
        with torch.no_grad():
            x = torch.cat([
                init_resources[i].unsqueeze(0),
                social_signals[i].detach().to(device)
            ], dim=-1)
            raw_logits = agent.trading_net(x.to(device)).to(device)
            positive_logits = F.softplus(raw_logits)
            w = project_to_simplex(positive_logits)
            u = torch.sum(w * torch.log(1.0 + current_resources[i]))
            initial_utilities.append(u)
            trading_weights.append(w)

    initial_utilities = torch.stack(initial_utilities)
    trading_weights = torch.stack(trading_weights)
    total_init = init_resources.sum(dim=0)

    for _ in range(pgd_steps):
        utilities = []
        for i, agent in enumerate(agents):
            x = torch.cat([
                current_resources[i].unsqueeze(0),
                social_signals[i].detach().to(device)
            ], dim=-1)
            raw_logits = agent.trading_net(x.to(device)).to(device)
            positive_logits = F.softplus(raw_logits)
            w = project_to_simplex(positive_logits)
            u = torch.sum(w * torch.log(1.0 + current_resources[i]))
            utilities.append(u)

        utilities = torch.stack(utilities)
        penalty = 100.0 * F.relu(initial_utilities - utilities).sum()
        loss = -utilities.sum() + penalty
        loss.backward()

        with torch.no_grad():
            current_resources -= pgd_lr * current_resources.grad
            current_resources = project_columns_onto_simplex(current_resources, total_init)
        current_resources.requires_grad_(True)

    with torch.no_grad():
        current_resources = project_columns_onto_simplex(current_resources, total_init)
    current_resources.grad = None

    return current_resources.detach().to(device)

def run_simulation():
    """Main training loop running agents through production, trade and consumption."""

    jobs = init_jobs()

    # Create separate optimizers for each subnetwork in each agent
    agents = [Agent().to(device) for _ in range(num_agents)]
    all_optimizers = [make_agent_optimizers(agent) for agent in agents]


    avg_reward = 0.0
    for _ in range(num_gif):
        for step in range(max_steps):

            avg_reward = 0.0
            for agent in agents:
                agent.resources = agent.resources.detach().to(device).requires_grad_(True)

            # Forward passes
            broadcasts = torch.stack([agent.broadcast_net(agent.resources) for agent in agents]).squeeze(1)
            broadcast_history.append(broadcasts.detach().cpu())
            social_signals = compute_social_signals(agents, broadcasts)

            # --- PRODUCTION ---
            chosen_jobs = []
            efforts = []
            produced = []
            inputs = []
            job_ids = []

            for i, agent in enumerate(agents):
                inp = torch.cat([agent.resources, social_signals[i]], dim=-1)
                logits = agent.production_net(inp).squeeze(0)
                job_logits, effort_logit = logits[:-1], logits[-1]
                effort = (1.0 - (agent.age / agent.max_age)) ** age_effort_penalty * torch.sigmoid(effort_logit)
                job_id = torch.argmax(job_logits).item()
                job = jobs[job_id]

                required_input = job['input'] * effort

                # Only compute output if the agent can afford the job input

                # Compute seasonal scaling
                seasonal_scale = torch.cos(torch.tensor(1.57 * (step / job['period']) + job['phase'], device=device))**2.0 if job['period'] != float("Inf") else 1.0
                scaled_output = (job['output'] * seasonal_scale + job['min'] * job['output'])

                # Check if full effort is affordable
                required_input = job['input'] * effort
                if torch.all(agent.resources >= required_input - 1e-6):
                    output = scaled_output * effort
                    chosen_jobs.append(job_id)
                    efforts.append(effort)
                    produced.append(output)
                    inputs.append(required_input)
                    job_ids.append(i)
                else:
                    # Compute max affordable effort
                    # Compute max affordable effort
                    with torch.no_grad():
                        resource_vec = agent.resources.squeeze(0)  # Shape [3]
                        input_vec = job['input']
                        denom = input_vec.clone()
                        denom[denom == 0] = 1e-8  # Avoid division by zero
                        effort_cap = resource_vec / denom  # Shape [3]
                        mask = input_vec > 0
                        if mask.sum() == 0:
                            max_effort = 0.0
                        else:
                            max_effort = torch.min(effort_cap[mask])

                        if max_effort > 1e-3:
                            output = scaled_output * max_effort
                            required_input = job['input'] * max_effort
                            chosen_jobs.append(job_id)
                            efforts.append(max_effort)
                            produced.append(output)
                            inputs.append(required_input)
                            job_ids.append(i)
                            agent.last_effort = max_effort.detach()
                        else:
                            agent.last_effort = torch.tensor(0.0, device=device)
            # --- PRODUCTION (no resource cap enforcement) ---
            for j, i in enumerate(job_ids):
                output = produced[j]
                #print(output)
                required_input = inputs[j]
                new_res = agents[i].resources - required_input + output
                new_res = torch.where(new_res.abs() < small_value_threshold, torch.zeros_like(new_res), new_res)
                agents[i].resources = new_res

                agents[i].last_effort = efforts[j].detach()

            #print(f"[STEP {step}] Resource totals BEFORE trading:",
            #torch.stack([agent.resources.squeeze(0) for agent in agents]).sum(dim=0))


            # --- TRADING ---


            try:
                before_trade = torch.stack([agent.resources.squeeze(0).detach().cpu() for agent in agents])
                final_resources = run_trading_pgd(agents, social_signals)
                after_trade = final_resources.detach().cpu()

                for i, agent in enumerate(agents):
                    res = final_resources[i].unsqueeze(0).to(device)
                    res = torch.where(res.abs() < small_value_threshold, torch.zeros_like(res), res)
                    agent.resources = res.detach().requires_grad_(True)
            except Exception as e:
                print(f"Trading failed at step {step}: {e}")
                for agent in agents:
                    agent.resources = agent.resources.detach().to(device).requires_grad_(True)
            #print(f"[STEP {step}] Resource totals AFTER trading:",
    #torch.stack([agent.resources.squeeze(0) for agent in agents]).sum(dim=0))

            # --- CONSUMPTION + REWARD ---
            optimizer_idx = 0
            for i, agent in enumerate(agents):
                # Forward pass through consumption net
                x = torch.cat([agent.resources, social_signals[i]], dim=-1)
                cf = torch.clamp(F.relu(agent.consumption_net(x)), max=0.95)
                consumed = agent.resources.detach() * cf
                agent.cached_outputs["last_consumed"] = consumed.detach()

                consumed = consumed.squeeze(0)

                # Shared rewardi
                reward = torch.prod((1.0+consumed)**agent.reward_weights.to(device)) + alpha*(1.0-agent.last_effort) - 1.0
                #reward = (reward_weights * (1 + consumed).log()).sum()
                #reward = torch.min(reward_weights*consumed)+alpha*(1.0-agent.last_effort)
                #reward = (consumed.squeeze(0) * reward_weights).sum() + alpha*(1.0 - agent.last_effort)#torch.exp((reward_weights * (consumed + 1e-8).log()).sum())#(consumed.squeeze(0) * reward_weights).sum() + alpha*(1.0 - agent.last_effort)
                relative_reward = reward - agent.reward_baseline
                agent.reward_baseline = 0.70 * agent.reward_baseline + 0.30 * reward.item()
                loss = -relative_reward
                agent.accum_reward += relative_reward
                avg_reward += relative_reward
                # Separate backward passes for each network
                for net, opt in zip([agent.broadcast_net, agent.social_filter_net,
                                        agent.production_net, agent.trading_net, agent.consumption_net],
                                        all_optimizers[i]):
                    opt.zero_grad()
                    loss.backward(retain_graph=True, inputs=list(net.parameters()))
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()

                # Decay and resource update
                resources_after = agent.resources.detach() * (1 - cf)
                leftover = resources_after
                decayed = leftover * decay_rate
                res = (leftover - decayed)
                res = torch.where(res.abs() < small_value_threshold, torch.zeros_like(res), res)
                agent.resources = res.detach().requires_grad_(True)
                agent.age += 1
            avg_reward /= num_agents
            resource_totals = torch.stack([agent.resources.sum() for agent in agents])
            gini_val = gini(resource_totals)
            gini_history.append(gini_val)
            agents, all_optimizers = prune_underconsuming_agents(agents, min_vector)
            resource_history.append([
                sum(agent.resources[0, r].item() for agent in agents)
                for r in range(input_dim)
            ])
            if step % plot_freq == 0 :
                job_names = [job['name'] for job in jobs]
                plot_trade_with_supply_demand(before_trade, after_trade, step, agents=agents,broadcasts=broadcasts, job_names=job_names, chosen_jobs=chosen_jobs)



            print(f"Step {step} Avg Reward: {avg_reward} Gini: {gini_val}")
    return agents

if __name__ == "__main__":
    agents = run_simulation()
    #generate_trade_gif()
    plt.plot(gini_history)
    plt.title("Gini Index of Resource Distribution Over Time")
    plt.xlabel("Step")
    plt.ylabel("Gini Coefficient")
    plt.savefig("gini_over_time.png")

