import json
import csv
import numpy as np

def generate_identical_twin_traffic_data():
    """
    Generates synthetic 'real-world' data for the 1D Mobile Agent (Section 6.8.4).
    Outputs metadata to JSON and the timeline data to CSV.
    """
    # 1. Define the "Hidden" True System Parameters
    system_params = {
        "dt": 1.0,
        "true_initial_position": 20.0,
        "true_initial_velocity": 10.0,
        "process_noise_pos_std": 1.5,
        "process_noise_vel_std": 0.5,
        "ambient_temp": 10.0,
        "fire_temp_rise": 30.0,
        "fire_spread_constant": 15.0,
        "sensor_noise_std": 2.0,
        "fireplace_1_pos": 80.0,
        "fireplace_2_pos": 200.0,
        "total_steps": 30
    }

    # Save parameters to JSON
    with open('data/true_system_case_1.json', 'w') as f:
        json.dump(system_params, f, indent=4)

    # 2. Simulate the "Real World"
    np.random.seed(42) # Fixed seed so the "Real World" is reproducible
    
    pos = system_params["true_initial_position"]
    vel = system_params["true_initial_velocity"]
    
    data_log =[]
    
    for k in range(1, system_params["total_steps"] + 1):
        # Apply True Process Noise (Equation 6.75)
        pos = pos + vel * system_params["dt"] + np.random.normal(0, system_params["process_noise_pos_std"])
        vel = vel + np.random.normal(0, system_params["process_noise_vel_std"])
        
        # Calculate True Temperature at current position
        t1 = system_params["fire_temp_rise"] * np.exp(-((pos - system_params["fireplace_1_pos"])**2) / (system_params["fire_spread_constant"]**2)) + system_params["ambient_temp"]
        t2 = system_params["fire_temp_rise"] * np.exp(-((pos - system_params["fireplace_2_pos"])**2) / (system_params["fire_spread_constant"]**2)) + system_params["ambient_temp"]
        true_temp = max(t1, t2)
        
        # Apply Sensor Noise (Equation 6.76)
        noisy_temp = true_temp + np.random.normal(0, system_params["sensor_noise_std"])
        
        # Log the step
        data_log.append({
            "time_step": k,
            "true_position": round(pos, 3),
            "true_velocity": round(vel, 3),
            "true_temperature": round(true_temp, 3),
            "noisy_sensor_reading": round(noisy_temp, 3)
        })

    # 3. Save to CSV
    with open('data/traffic_ground_truth.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_log[0].keys())
        writer.writeheader()
        writer.writerows(data_log)
        
    print("Generated data/traffic_ground_truth.csv and data/true_system_case_1.json")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    generate_identical_twin_traffic_data()