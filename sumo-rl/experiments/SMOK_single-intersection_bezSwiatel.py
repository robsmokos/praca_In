import argparse
import os
import sys
from datetime import datetime
import csv



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("c:\\Users\\rob\\Sumo")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route", 
        dest="route",
        type=str,
        default= "c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\nets\\SMOK_single-intersection_bezSwiatel\\single-intersection.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=0, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=0, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=10000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\outputs\\SMOK_single-intersection_bezSwiatel\\alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"

    env = SumoEnvironment(
        net_file="c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\nets\\SMOK_single-intersection_bezSwiatel\\single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    with open('c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\outputs\\SMOK_single-intersection_bezSwiatel\\traffic_light_actions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'time', 'c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\outputs\\SMOK_single-intersection_bezSwiatel\\traffic_light_id', 'action'])

        for run in range(1, args.runs + 1):
            initial_states = env.reset()
            ql_agents = {
                ts: QLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    exploration_strategy=EpsilonGreedy(
                        initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay
                    ),
                )
                for ts in env.ts_ids
            }

            done = {"__all__": False}
            infos = []
            if args.fixed:
                while not done["__all__"]:
                    _, _, done, _ = env.step({})
            else:
                while not done["__all__"]:
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                    
                    ## wylacz zmienianie swiatel
                    ## actions = {ts: 1 for ts in actions.keys()}
                    
                    # Logowanie akcji do pliku CSV
                    for ts, action in actions.items():
                        writer.writerow([run, env.sim_step, ts, action])
               
                    s, r, done, _ = env.step(action=actions)

                    for agent_id in ql_agents.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                    
            env.save_csv(out_csv, run)
            env.close()
