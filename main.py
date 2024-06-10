from environment.environment import Environment
import numpy as np

def main():
    np.random.seed(0)
    # Environment initializes correctly with valid parameters
    number_of_servers = 3  # Changed from 10 to 3
    private_cpu_capacities = [10 for i in range(1, number_of_servers + 1)]
    public_cpu_capacities = [15.0 for i in range(1, number_of_servers + 1)]
    episode_time = 100
    connection_matrix = np.array([[0 if i == j else 100 * (i + 1) for j in range(number_of_servers)] for i in range(number_of_servers)])
    cloud_computational_capacity = 10000.0
    task_arrive_probabilities = [1] * number_of_servers
    task_size_mins = [1 for i in range(1, number_of_servers + 1)]
    task_size_maxs = [10 for i in range(1, number_of_servers + 1)]
    task_size_distributions = ['uniform'] * number_of_servers
    timeout_delay_mins = [1 for i in range(1, number_of_servers + 1)]
    timeout_delay_maxs = [10 * i for i in range(1, number_of_servers + 1)]
    timeout_delay_distributions = ['choice'] * number_of_servers
    priotiry_mins = [1 for i in range(1, number_of_servers + 1)]
    priotiry_maxs = [10 for i in range(1, number_of_servers + 1)]
    priotiry_distributions = ['choice'] * number_of_servers
    computational_density_mins = [1 for i in range(1, number_of_servers + 1)]
    computational_density_maxs = [10 for i in range(1, number_of_servers + 1)]
    computational_density_distributions = ['uniform'] * number_of_servers
    drop_penalty_mins = [1 for i in range(1, number_of_servers + 1)]
    drop_penalty_maxs = [10 for i in range(1, number_of_servers + 1)]
    drop_penalty_distributions = ['choice'] * number_of_servers

    env = Environment(
        number_of_servers,
        private_cpu_capacities,
        public_cpu_capacities,
        connection_matrix,
        cloud_computational_capacity,
        episode_time,
        task_arrive_probabilities,
        task_size_mins,
        task_size_maxs,
        task_size_distributions,
        timeout_delay_mins,
        timeout_delay_maxs,
        timeout_delay_distributions,
        priotiry_mins,
        priotiry_maxs,
        priotiry_distributions,
        computational_density_mins,
        computational_density_maxs,
        computational_density_distributions,
        drop_penalty_mins,
        drop_penalty_maxs,
        drop_penalty_distributions
    )

    assert env.number_of_servers == number_of_servers
    assert len(env.servers) == number_of_servers
    assert env.cloud.computational_capacity == cloud_computational_capacity

    for i in range(100):
        # Choose randomly between 0 and 1
        actions = np.random.randint(number_of_servers, size=number_of_servers)
        # Assuming you want to use the action in some way, e.g., env.step
        env.step(actions)

if __name__ == "__main__":
    main()