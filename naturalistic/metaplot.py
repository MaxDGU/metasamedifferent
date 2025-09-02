import matplotlib.pyplot as plt
import numpy as np

# Initial meta-parameters
theta_0 = np.array([0.5, 2.5])

# Hypothetical optimal parameters for each support example
# These are chosen arbitrarily to show distinct pulls.
# In reality, these are not explicit points but define directions of gradients.
P_s1_star = np.array([2.0, 2.0])  # Optimal for Support 1
P_s2_star = np.array([1.0, 0.5])  # Optimal for Support 2
P_s3_star = np.array([3.0, 1.0])  # Optimal for Support 3
P_s4_star = np.array([2.5, 3.0])  # Optimal for Support 4

# Inner loop learning rate (alpha)
alpha = 0.4  # Learning rate for inner loop updates

# --- Simulate Inner Loop Gradient Steps ---
parameters_trajectory = [theta_0]
current_theta = theta_0.copy()

# Step 1: Update for Support Example 1
grad_s1 = current_theta - P_s1_star  # Conceptual gradient (points from P_s1_star to current_theta)
theta_1_prime = current_theta - alpha * grad_s1 # Gradient descent step
parameters_trajectory.append(theta_1_prime)
current_theta = theta_1_prime.copy()

# Step 2: Update for Support Example 2
grad_s2 = current_theta - P_s2_star
theta_2_prime = current_theta - alpha * grad_s2
parameters_trajectory.append(theta_2_prime)
current_theta = theta_2_prime.copy()

# Step 3: Update for Support Example 3
grad_s3 = current_theta - P_s3_star
theta_3_prime = current_theta - alpha * grad_s3
parameters_trajectory.append(theta_3_prime)
current_theta = theta_3_prime.copy()

# Step 4: Update for Support Example 4
grad_s4 = current_theta - P_s4_star
theta_4_prime = current_theta - alpha * grad_s4
parameters_trajectory.append(theta_4_prime)

# Convert trajectory to numpy array for easier plotting
parameters_trajectory = np.array(parameters_trajectory)

# --- Plotting ---
plt.figure(figsize=(10, 8))

# Plot hypothetical optimal points for each support example
plt.scatter(P_s1_star[0], P_s1_star[1], marker='*', s=200, color='red', label='Optimal for S1 ($P^*_{S1}$)', zorder=5)
plt.scatter(P_s2_star[0], P_s2_star[1], marker='*', s=200, color='green', label='Optimal for S2 ($P^*_{S2}$)', zorder=5)
plt.scatter(P_s3_star[0], P_s3_star[1], marker='*', s=200, color='blue', label='Optimal for S3 ($P^*_{S3}$)', zorder=5)
plt.scatter(P_s4_star[0], P_s4_star[1], marker='*', s=200, color='purple', label='Optimal for S4 ($P^*_{S4}$)', zorder=5)

# Plot parameter trajectory
plt.plot(parameters_trajectory[:, 0], parameters_trajectory[:, 1], marker='o', linestyle='-', color='black', label='Parameter Trajectory')

# Annotate points
point_labels = [r'$\theta_0$ (Initial)', r"$\theta'_1$ (after S1)", r"$\theta'_2$ (after S2)", r"$\theta'_3$ (after S3)", r"$\theta'_4 = \phi_t$ (after S4)"]
for i, label in enumerate(point_labels):
    plt.text(parameters_trajectory[i, 0] + 0.05, parameters_trajectory[i, 1] + 0.05, label, fontsize=12)
    plt.scatter(parameters_trajectory[i, 0], parameters_trajectory[i, 1], s=100, color='black', zorder=5)


# Add arrows for gradient steps
for i in range(len(parameters_trajectory) - 1):
    start_point = parameters_trajectory[i]
    end_point = parameters_trajectory[i+1]
    arrow_color = ['red', 'green', 'blue', 'purple'][i] # Color matches the target optimal point
    plt.annotate("",
                 xy=end_point, xycoords='data',
                 xytext=start_point, textcoords='data',
                 arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, shrinkA=5, shrinkB=5),
                 )
    # Add gradient labels
    mid_point = (start_point + end_point) / 2
    plt.text(mid_point[0] + 0.1, mid_point[1] - 0.1, fr'$-\alpha \nabla L_{{S{i+1}}}$', fontsize=11, color=arrow_color)


# Style the plot
plt.title('MAML Inner Loop: Gradient Steps on Support Set (Conceptual)', fontsize=16)
plt.xlabel('Parameter 1', fontsize=14)
plt.ylabel('Parameter 2', fontsize=14)
plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.xlim(-0.5, 3.5)
plt.ylim(0, 3.5)
plt.gca().set_aspect('equal', adjustable='box') # Ensure arrows are not distorted if scales differ
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

plt.show()