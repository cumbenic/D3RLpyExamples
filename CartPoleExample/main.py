import gym
from d3rlpy.algos import CQL
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment

# Load CartPole environment data
dataset, env = get_cartpole()

# Create and configure the CQL algorithm
algo = CQL(n_epochs=100, q_func_factory='qrnn', use_gpu=False)

# Train the algorithm on the dataset
algo.fit(dataset)

# Evaluate the agent's performance
score = evaluate_on_environment(algo, env)
print(f'Score: {score}')

# Save the trained model
algo.save_model('cql_model.pkl')
