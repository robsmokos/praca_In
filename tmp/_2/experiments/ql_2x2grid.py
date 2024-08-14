import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("c:\\Users\\rob\\Sumo")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
#    alpha = 0.1
    alpha = 0.1
    
    gamma = 0.99
    decay = 1
    runs = 30
    episodes = 4

    env = SumoEnvironment(
        net_file="c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\sumo_rl\\nets\\2x2grid\\2x2.net.xml",
        route_file="c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\sumo_rl\\nets\\2x2grid\\2x2.rou.xml",
        use_gui=False,
        num_seconds=800,
        min_green=5,
        delta_time=5,
        
        
        
    )

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

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}


                #for ts_id, traffic_signal in env.traffic_signals.items():
                   # print("Agent ID:", ts_id )  # Wyświetlenie ID agenta
                   # print("Current Green Phase:", traffic_signal.green_phase)  # Wyświetlenie aktualnej fazy zielonej
                   # print("Is Yellow Phase:", traffic_signal.is_yellow)  # Wyświetlenie informacji czy jest faza żółta
                   # print("Time Until Next Action:", traffic_signal.next_action_time - env.sim_step)  # Wyświetlenie czasu do następnej akcji
                   # print("Time Since Last Phase Change:", traffic_signal.time_since_last_phase_change)  # Wyświetlenie czasu od ostatniej zmiany fazy

                     #print("ID:", ts_id, "Current Green Ph:", traffic_signal.green_phase, "Time Since Last Phase Change:", traffic_signal.time_since_last_phase_change)

                
                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
       
        
                for agent_id, reward in r.items():
                    print(f"Agent {agent_id} Current Reward: {reward}")
            
            env.save_csv(f"c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\outputs\\2x2\\ql-2x2grid_run{run}", episode)
            
    
              
            
            
    env.close()
