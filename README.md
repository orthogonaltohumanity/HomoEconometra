# HomoEconometra

**HomoEconometra** is an experimental agent-based simulation designed to explore the emergence of decentralized economic behavior, communication, and social specialization in a simplified artificial society. The project combines neural reinforcement learning, constrained optimization, and dynamic signaling to simulate trade, survival, and cultural differentiation.

---

## Overview

In HomoEconometra, a population of autonomous agents navigates a minimal economy governed by:

* Basic resources (e.g., Berries and Meat)
* Seasonal and stochastic job structures
* Trade governed by constrained optimization (projected gradient descent)
* Learned communication and selective attention
* Consumption-based fitness and survival
* Evolutionary reproduction based on accumulated reward

Each agent is implemented as a multi-network learning system, with distinct modules for communication, social evaluation, production, trading, and consumption.

---

## Simulation Cycle

Each simulation step proceeds as follows:

1. **Broadcast**: Each agent emits a vector signal derived from its internal state.
2. **Social Filtering**: Agents evaluate the signals of others using a learned attention mechanism.
3. **Production**: Agents choose a job and effort level based on their local context.
4. **Trading**: Resources are reallocated using projected gradient descent over utility functions.
5. **Consumption**: Agents convert resources into reward, guided by inherited utility weights.
6. **Reproduction and Decay**: Underperforming or aging agents are replaced by offspring.
7. **Logging and Visualization**: Key state variables and structures are saved for analysis.

---

## Agent Architecture

Each agent contains five independently trained neural networks:

* `broadcast_net`: Maps internal resource state to a public signal vector.
* `social_filter_net`: Assigns weights to other agents’ broadcasts based on pairwise features.
* `production_net`: Selects job type and effort level from combined resource and signal input.
* `trading_net`: Determines preferences over resources for trade optimization.
* `consumption_net`: Specifies resource consumption ratios to maximize reward.

Each agent also carries a heritable reward weight vector, which encodes its subjective utility function.

---

## Features

* Independent reinforcement learning for each agent's submodules
* Learned communication system with end-to-end differentiable attention
* Multi-good trading via constrained gradient-based optimization
* Emergent social behavior: specialization, inequality, influence
* High-resolution visual diagnostics per time step

---

## Installation

This project requires Python 3.8+ and PyTorch.

```bash
git clone https://github.com/yourusername/homoeconometra.git
cd homoeconometra
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python main.py
```

This will:

* Simulate agents for `max_steps` iterations
* Save plots in the `trade_scatter/` directory
* Generate an animated `.gif` of trade and role evolution
* Output Gini inequality metrics over time

---

## Output Artifacts

Each time step logs:

* Resource totals
* Agent ages and reproduction statistics
* Job distributions
* Broadcast vectors (visualized using PCA)
* Social attention weights (heatmaps)
* Eigenvalue spectra of the communication space
* Pre- and post-trade resource allocations


---

## Research Questions

* How do agents allocate social attention over time?
* Do communication vectors converge to low-dimensional semantic structures?
* What mechanisms drive emergent inequality and specialization?
* Does social filtering improve reward outcomes relative to uniform interactions?
* How does agent heterogeneity affect economic dynamics?

---

## Future Directions

* Extension to multi-layer supply chains and compound goods
* Structured or symbolic communication systems
* Spatial simulation with migration and territory control
* Genealogical tracking and cultural inheritance
* Integration with external language models or theorem provers

---

## License

MIT License. Open for academic and research use.

---

## Author

Mira Samantha Lanier Kennard
Independent researcher in artificial economies and emergent coordination systems
