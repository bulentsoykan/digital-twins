import csv
import numpy as np

def generate_wildfire_sensor_data():
    """
    Generates synthetic sensor readings for a spreading wildfire (Section 8.4).
    Places 10 stationary sensors on a 50x50 grid and logs their noisy temperatures over time.
    """
    GRID_SIZE = 50
    TOTAL_STEPS = 20
    NUM_SENSORS = 10
    
    # 1. Randomly deploy 10 stationary sensors (Random Deployment Schema 1)
    np.random.seed(99)
    sensors =[]
    for s_id in range(NUM_SENSORS):
        sensors.append({
            "sensor_id": f"S_{s_id}",
            "x": np.random.randint(5, 45),
            "y": np.random.randint(5, 45)
        })

    # 2. Simulate fire spread and sensor readings
    data_log =[]
    
    # Fire starts in the center of the grid
    fire_center_x, fire_center_y = 25, 25
    fire_radius = 2.0 # Initial fire size
    
    for step in range(TOTAL_STEPS):
        # Fire spreads outward by roughly 1.5 units per time step + some wind noise
        fire_radius += 1.5 + np.random.normal(0, 0.2)
        
        # Read temperatures from all sensors
        for sensor in sensors:
            # Calculate distance from sensor to the closest point on the fire ring
            dist_to_center = np.sqrt((sensor["x"] - fire_center_x)**2 + (sensor["y"] - fire_center_y)**2)
            dist_to_fire_front = abs(dist_to_center - fire_radius)
            
            # If fire has passed the sensor, distance is effectively 0 (it is in the burned area)
            if dist_to_center < fire_radius:
                dist_to_fire_front = 0.0
            
            # Equation 8.7: T = T_c * exp(-d^2 / 2*sigma^2) + T_a
            T_a = 25.0  # Ambient 25C
            T_c = 400.0 # Fire 400C
            sigma = 5.0 # Heat dissipation
            
            true_temp = T_c * np.exp(-(dist_to_fire_front**2) / (2 * sigma**2)) + T_a
            
            # Add Gaussian sensor noise
            noisy_temp = true_temp + np.random.normal(0, 3.0) # 3 degree sensor error
            
            data_log.append({
                "time_step": step,
                "sensor_id": sensor["sensor_id"],
                "sensor_x": sensor["x"],
                "sensor_y": sensor["y"],
                "true_fire_radius": round(fire_radius, 2),
                "noisy_temp_reading": round(noisy_temp, 2)
            })

    # 3. Save to CSV
    with open('data/wildfire_sensor_readings.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_log[0].keys())
        writer.writeheader()
        writer.writerows(data_log)

    print("Generated data/wildfire_sensor_readings.csv")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    generate_wildfire_sensor_data()