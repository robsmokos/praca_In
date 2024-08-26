import os
import sys
import csv

# Ensure SUMO_HOME is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 1

    env = SumoEnvironment(
        net_file="test_flow.net.xml",
        route_file="test_flow.rou.xml",
        use_gui=False,
        num_seconds=10000,
        min_green=5,
        delta_time=5,
    )

    # Prepare the CSV file for logging actions
    csv_file_path = 'traffic_light_actions.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'time', 'traffic_light_id', 'action'])

        for run in range(1, runs + 1):
            initial_states = env.reset()
            ql_agents = {
                ts: QLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
                )
                for ts in env.ts_ids
            }

            for episode in range(1, episodes + 1):
                if episode != 1:
                    initial_states = env.reset()
                    for ts in initial_states.keys():
                        ql_agents[ts].state = env.encode(initial_states[ts], ts)

                done = {"__all__": False}
                while not done["__all__"]:
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                    # Log actions to the CSV file
                    current_time = env.sim_step  # Assuming sim_step gives the current simulation time
                    for ts, action in actions.items():
                        writer.writerow([run, current_time, ts, action])
                        
                        
                        # Log actions to CSV
                    for ts, action in actions.items():
                        writer.writerow([run, env.sim_step, ts, action])    
                        
                    s, r, done, info = env.step(action=actions)

                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                env.save_csv(f"choinka_rl{run}", episode)

    env.close()
