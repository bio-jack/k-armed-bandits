import numpy as np
import matplotlib.pyplot as plt

# Initialize number of arms on k-Armed bandit and associated true values
k = 10
trueValues = np.random.default_rng().normal(size=k)
# Keep record of the optimal action unknown to the agent
optimalArm = np.argmax(trueValues)

def pull(arm):
  """
  Simulates pulling an arm on the k-Armed bandit and returns the associated 
  reward.

  Parameters
  ----------
  arm (int)
    Integer associated with arm on k-Armed bandit.
  
  Returns
  ----------
  reward (float)
    Reward associated with pulling bandit arm. Drawn from a normal distribution,
    mean = the true value of the arm
    
  """
  reward = np.random.default_rng().normal(loc=trueValues[arm])
  return reward

def epsilonGreedy(e):
  """
  Implement e-greedy action selection for the k-Armed bandit problem.

  Parameters
  ----------
  e (float)
    Epsilon value - probability of selecting exploratory action

  Returns
  ----------
  data (np.array)
    Reward accumulated at each timestep over 1000 timesteps, averaged over 2000 
    runs.

  """
  # Initialise list of final cumulative rewards + optimal action selections
  data = np.zeros(1000)
  optimalSelection = np.zeros(1000)
  
  # Iterate over 2000 runs
  for _ in range(2000):
    
    # Initialise value estimates for each arm
    valueEstimates = np.zeros(k)
    
    # Track number of pulls of each arm
    numPulls = np.zeros(k)
    timestep = 0
    
    # Track reward at each timestep 
    rewardAtTimestep = np.zeros(1000)

    # For 1000 timesteps / pulls
    for _ in range(1000):
      
      # Greedy action selection
      if np.random.random() > e:
        # Find arm with highest value estimate and pull (with tiebreak)
        arm = np.random.choice(np.where(valueEstimates[:] == valueEstimates[:].max())[0])
        reward = pull(arm)
        # Update value estimates and number of pulls for arm
        valueEstimates[arm] = valueEstimates[arm] + (1/(numPulls[arm]+1))*(reward - valueEstimates[arm])
        numPulls[arm] += 1
      
      # Exploratory action selection
      else:
        # Randomly select an arm and pull
        arm = np.random.randint(0,10)
        reward = pull(arm)
        # Update value estimates for arm and number of pulls for arm
        valueEstimates[arm] = valueEstimates[arm] + (1/(numPulls[arm]+1))*(reward-valueEstimates[arm])
        numPulls[arm] += 1
      
      if arm == optimalArm:
        optimalSelection[timestep] += 1
      
      # Update reward at step
      rewardAtTimestep[timestep] += reward
      timestep += 1

    # Add total reward at each timestep in the current run to the total data (for 2000 iterations)
    data += rewardAtTimestep

  # Average after 2000 runs
  data = data / 2000
  optimalSelection = optimalSelection / 2000
  return data, optimalSelection

# Get data for e = 0, 0.1, 0.01
output_0 = epsilonGreedy(0)
output_01 = epsilonGreedy(0.1)
output_001 = epsilonGreedy(0.01)

# Plot cumulative reward
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel("Timestep (t)")
ax.set_ylabel("Average reward")
ax.set_ylim([0, 1.5])
ax.plot(output_0[0], label="e = 0")
ax.plot(output_01[0], label="e = 0.1")
ax.plot(output_001[0], label = "e = 0.01")
fig.legend()

# Plot optimal action selection
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.set_xlabel("Timestep (t)")
ax2.set_ylabel("Fraction optimal action selection")
ax2.set_ylim([0, 1])
ax2.plot(output_0[1], label="e = 0")
ax2.plot(output_01[1], label="e = 0.1")
ax2.plot(output_001[1], label = "e = 0.01")
fig2.legend()

# Show figures
plt.show()
