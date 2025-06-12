import numpy as np 

def noise_normalize(x, min_val=0, max_val=100):
    return (x) / (max_val - min_val)

def make_guassion_noise(states, noise_scale):
    for state in states:
        for key in state.keys():
            if key == 'cur_phase' or key == 'time_this_phase' or key == 'adjacency_matrix':
                continue 
            else:
                if key == 'lane_enter_running_part':
                    noise_scale_normalized = noise_normalize(noise_scale, 0, 30)
                elif key == 'traffic_movement_pressure_queue_efficient':
                    noise_scale_normalized = noise_normalize(noise_scale, -1, 5)
                elif key == 'lane_num_waiting_vehicle_in':
                    noise_scale_normalized = noise_normalize(noise_scale, 0, 100)

                state[key] = (state[key] + noise_scale_normalized * np.random.randn(len(state[key]))).tolist()
    return states


def make_U_rand_noise(states, noise_scale):
    for state in states:
        for key in state.keys():
            if key == 'cur_phase' or key == 'time_this_phase' or key == 'adjacency_matrix':
                continue 
            else:
                if key == 'lane_enter_running_part':
                    noise_scale_normalized = noise_normalize(noise_scale, 0, 30)
                elif key == 'traffic_movement_pressure_queue_efficient':
                    noise_scale_normalized = noise_normalize(noise_scale, -1, 5)
                elif key == 'lane_num_waiting_vehicle_in':
                    noise_scale_normalized = noise_normalize(noise_scale, 0, 100)
                state[key] = (state[key] + 2 * noise_scale_normalized * np.random.randn(len(state[key]))- noise_scale_normalized).tolist()
    return states


