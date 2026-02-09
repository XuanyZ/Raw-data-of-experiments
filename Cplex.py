import sys
import pandas as pd
import glob
import os
import numpy as np
from docplex.mp.model import Model

DISTANCE_FILE = 'distance_matrix.csv'  # distance matrix
# TASKS_FILE = "2023-02-27.csv"   # task file


def load_data(dist_path, tasks_path):
    print(f"Loading distance matrix from: {dist_path}")
    df_dist = pd.read_csv(dist_path, header=None)
    dist_matrix = df_dist.values.tolist()

    print(f"Loading tasks from: {tasks_path}")
    df_tasks = pd.read_csv(tasks_path)

    try:
        tasks = []
        for index, row in df_tasks.iterrows():
            task_tuple = (
                int(row['0']),  # lp
                float(row['1']),  # tp
                int(row['2']),  # ld
                float(row['3']),  # td
                float(row['4'])  # ci
            )
            tasks.append(task_tuple)

    except KeyError as e:
        print(f"Error: Column {e} not found in tasks CSV. Please update the column names in the code.")
        sys.exit(1)

    return dist_matrix, tasks


def solve_ev_scheduling_dynamic(tasks, dist_matrix, stations, params):
    num_tasks = len(tasks)
    v = params['v']
    B = params['B']
    p = params['p']
    q = params['q']  # Time per unit energy
    bl = params['bl']

    pickups = [t[0] for t in tasks]
    starts = [t[1] for t in tasks]
    drops = [t[2] for t in tasks]
    ends = [t[3] for t in tasks]
    cons = [t[4] for t in tasks]

    def get_travel(loc1, loc2):
        dist = dist_matrix[loc1][loc2]
        time = dist / v
        energy = dist * bl
        return time, energy

    print("Building graph with Greedy Station Strategy...")

    arcs_direct = []
    arcs_charge = []

    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j: continue

            t_ij, e_ij = get_travel(drops[i], pickups[j])
            if ends[i] + t_ij <= starts[j]:
                arcs_direct.append((i, j, e_ij, t_ij))

            best_s = None
            min_detour_travel_time = float('inf')
            best_travel_metrics = (0, 0, 0, 0)  # (t1, e1, t2, e2)

            for s in stations:
                t1, e1 = get_travel(drops[i], s)
                t2, e2 = get_travel(s, pickups[j])

                if t1 + t2 < min_detour_travel_time:
                    min_detour_travel_time = t1 + t2
                    best_s = s
                    best_travel_metrics = (t1, e1, t2, e2)

            if best_s is not None:
                t1, e1, t2, e2 = best_travel_metrics
                if ends[i] + t1 + t2 <= starts[j]:
                    arcs_charge.append((i, j, best_s, t1, e1, t2, e2))

    print(f"Graph: {len(arcs_direct)} direct, {len(arcs_charge)} charged arcs.")

    # --- CPLEX Model ---
    mdl = Model('EV_Dynamic_Charge')
    M = 2 * B  # Big-M for logic constraints
    T_M = 100000  # Big-M for Time constraints (should be > max time horizon)

    # Variables
    # x_d: Direct travel i -> j
    x_d = mdl.binary_var_dict([(i, j) for i, j, e, t in arcs_direct], name='x_d')

    # x_c: Travel i -> Station -> j
    x_c = mdl.binary_var_dict([(i, j) for i, j, s, t1, e1, t2, e2 in arcs_charge], name='x_c')

    # y[i]: Battery level at start of task i
    y = mdl.continuous_var_dict(range(num_tasks), lb=p, ub=B, name='y')

    # is_start[i]: Vehicle starts here
    is_start = mdl.binary_var_dict(range(num_tasks), name='is_start')

    # Objective
    mdl.minimize(mdl.sum(is_start[i] for i in range(num_tasks)))

    # --- Constraints ---
    for j in range(num_tasks):
        in_d = mdl.sum(x_d[i, j] for i, j_dummy, e, t in arcs_direct if j_dummy == j)
        in_c = mdl.sum(x_c[i, j] for i, j_dummy, s, t1, e1, t2, e2 in arcs_charge if j_dummy == j)

        mdl.add_constraint(is_start[j] + in_d + in_c == 1, f"FlowIn_{j}")

        mdl.add_constraint(y[j] >= B * is_start[j], f"InitBat_{j}")

    for i in range(num_tasks):
        out_d = mdl.sum(x_d[i, j_dummy] for i_dummy, j_dummy, e, t in arcs_direct if i_dummy == i)
        out_c = mdl.sum(x_c[i, j_dummy] for i_dummy, j_dummy, s, t1, e1, t2, e2 in arcs_charge if i_dummy == i)

        mdl.add_constraint(out_d + out_c <= 1, f"FlowOut_{i}")
        mdl.add_constraint(y[i] - cons[i] >= p, f"FeasibleTask_{i}")

    for i, j, e_travel, t_travel in arcs_direct:
        mdl.add_constraint(y[j] <= y[i] - cons[i] - e_travel + M * (1 - x_d[i, j]), f"BatDir_{i}_{j}")

    for i, j, s, t1, e1, t2, e2 in arcs_charge:
        start_level_j = B - e2
        mdl.add_constraint(y[j] <= start_level_j + M * (1 - x_c[i, j]), f"BatChg_{i}_{j}")

        rhs_val = ends[i] + t1 + t2 + q * (B + cons[i] + e1) - starts[j]

        mdl.add_constraint(
            q * y[i] + T_M * (1 - x_c[i, j]) >= rhs_val,
            f"TimeChg_{i}_{j}"
        )

        mdl.add_constraint(
            y[i] - cons[i] - e1 + M * (1 - x_c[i, j]) >= 0,
            f"ReachSt_{i}_{j}"
        )

    # --- Solve ---
    mdl.parameters.timelimit = 60
    solution = mdl.solve(log_output=True)

    if solution:
        print(f"\n>>> Optimal Vehicles: {int(solution.objective_value)}")
    else:
        print("No feasible solution.")


if __name__ == "__main__":

    folder_path = 'small-scale-instances'  # folder path
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    sorted_files = sorted(csv_files)
    for TASKS_FILE in sorted_files:
        dist_matrix, tasks = load_data(DISTANCE_FILE, TASKS_FILE)

        params = {
            'v': 7,
            'B': 100,
            'p': 20.0,
            'q': 0.33,
            'bl': 0.002
        }

        stations = [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                    79, 80]

        solve_ev_scheduling_dynamic(tasks, dist_matrix, stations, params)
