import argparse
import os
import sys
from datetime import datetime
import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def log_simulation_data(file_path):
    """Log simulation data to a file."""
    with open(file_path, 'w') as f:
        # Logging traffic light data
        for traffic_light_id in traci.trafficlight.getIDList():
            phase = traci.trafficlight.getPhase(traffic_light_id)
            phase_duration = traci.trafficlight.getPhaseDuration(traffic_light_id)
            f.write(f"{traffic_light_id}, Phase: {phase}, Duration: {phase_duration} seconds\n")

        # Logging vehicle data
        for vehicle_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(vehicle_id)
            position = traci.vehicle.getPosition(vehicle_id)
            f.write(f"{vehicle_id}, Speed: {speed}, Position: {position}\n")

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route", 
        dest="route",
        type=str,
        default="c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\sumo_rl\\nets\\single-intersection\\single-intersection.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=True, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=500, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=50, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\outputs\\single-intersection\\alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"
    simulation_data_log_file = f"c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\outputs\\single-intersection\\simulation_data_log_{experiment_time}.txt"

    # Start SUMO simulation
    traci.start(["sumo-gui", "-c", "your_config_file.sumocfg"])  # Replace with your SUMO configuration file

    env = SumoEnvironment(
        net_file="c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\sumo_rl\\nets\\single-intersection\\single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

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
                s, r, done, _ = env.step(action=actions)
                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        # Log simulation data
        log_simulation_data(simulation_data_log_file)

        env.save_csv(out_csv, run)
        env.close()
    
    traci.close()
