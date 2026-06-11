import numpy as np
import matplotlib.pyplot as plt
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass


@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.5
    cohesion_weight: float = 0.1
    separation_weight: float = 0.5
    max_velocity = Vector2(1, 1)
    delta_time: float = 0.5
    mass: int = 20

    def weights(self) -> tuple[float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)


class Bird(Agent):
    config: FlockingConfig

    def change_position(self):
        a, c, s = self.config.weights()
        mass = self.config.mass

        agents = list(self.in_proximity_accuracy())
        agents_total_velocity = Vector2(0, 0)
        separation_vector = Vector2(0, 0)
        average_position = Vector2(0, 0)

        if len(agents) > 0:
            for agent, dist in agents:
                agents_total_velocity += agent.move
                separation_vector += (self.pos - agent.pos) / dist
                average_position += agent.pos

            average_velocity = agents_total_velocity / len(agents)
            average_position /= len(agents)
            cohesion_force = average_position - self.pos

            alignment = (average_velocity - self.move)
            separation = separation_vector / len(agents)
            cohesion = cohesion_force

        else:
            alignment = Vector2(0, 0)
            separation = Vector2(0, 0)
            cohesion = Vector2(0, 0)

        ftotal = (a * alignment + c * cohesion + s * separation) / mass

        self.move += ftotal

        if self.move.length() > self.config.max_velocity.length():
            self.move.scale_to_length(self.config.max_velocity.length())

        self.pos += self.move * self.config.delta_time
        self.there_is_no_escape()


def average_nn_distance(simulation):
    """Compute the average nearest-neighbour distance across all agents."""
    agents = list(simulation._agents)
    if len(agents) < 2:
        return 0.0
    positions = [agent.pos for agent in agents]
    distances = []
    for i, pos in enumerate(positions):
        others = [p for j, p in enumerate(positions) if j != i]
        nn_dist = min(pos.distance_to(p) for p in others)
        distances.append(nn_dist)
    return float(np.mean(distances))


# --- Experiment Settings ---
SEPARATION_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]  # Conditions to test
SEEDS = list(range(10))                           # 10 runs per condition
TOTAL_TICKS = 500                                 # Total ticks per run
STABILISATION_TICKS = 200                         # Ignore first 200 ticks
# Results are averaged over ticks 200-500 (the stable phase)

# --- Run Experiment ---
results = {}  # separation_weight -> list of mean nn distances (one per seed)

for sep_w in SEPARATION_VALUES:
    results[sep_w] = []
    for seed in SEEDS:
        simulation = Simulation(
            FlockingConfig(
                image_rotation=True,
                movement_speed=1,
                radius=50,
                seed=seed,
                separation_weight=sep_w,
            )
        )
        simulation.batch_spawn_agents(50, Bird, images=["images/white.png"])

        tick_distances = []

        for tick in range(TOTAL_TICKS):
            simulation.tick()
            if tick >= STABILISATION_TICKS:
                tick_distances.append(average_nn_distance(simulation))

        run_mean = float(np.mean(tick_distances))
        results[sep_w].append(run_mean)
        print(
            f"sep_weight={sep_w:.1f} | seed={seed} | mean_nn_dist={run_mean:.2f}")

# --- Compute Summary Statistics ---
means = [np.mean(results[s]) for s in SEPARATION_VALUES]
stds = [np.std(results[s]) for s in SEPARATION_VALUES]

# --- Plot Results ---
plt.figure(figsize=(8, 5))
plt.errorbar(SEPARATION_VALUES, means, yerr=stds,
             fmt='o-', capsize=5, color='steelblue')
plt.xlabel("Separation Weight")
plt.ylabel("Average Nearest-Neighbour Distance (px)")
plt.title("Effect of Separation Weight on Flock Cohesion")
plt.xticks(SEPARATION_VALUES)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("results_plot.png", dpi=150)
plt.show()
print("Plot saved as results_plot.png")
