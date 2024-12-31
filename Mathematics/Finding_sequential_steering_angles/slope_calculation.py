import numpy as np
import matplotlib.pyplot as plt

##Example non-intersecting curves

def generate_curves(start, end, num_points):
    x = np.linspace(start, end, num_points)
    curve1_y = np.sin(x) + 2  
    curve2_y = np.cos(x) - 2  
    return x, curve1_y, curve2_y

def calculate_slope(x, y, idx):
    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    return dy / dx

def calculate_steering_angle(tangent_slope, base_angle=0):
    return np.degrees(np.arctan(tangent_slope - base_angle))

##For the plot 
start = 0
end = 10
num_points = 100

x, curve1_y, curve2_y = generate_curves(start, end, num_points)

##Implementation
steering_angles = []
for i in range(1, num_points - 1):
    slope1 = calculate_slope(x, curve1_y, i)
    slope2 = calculate_slope(x, curve2_y, i)
    
    steering_angle = calculate_steering_angle(slope1)
    steering_angles.append(steering_angle)

    steering_angle2 = calculate_steering_angle(slope2)  
    steering_angles.append(steering_angle2)

plt.figure(figsize=(10, 6))
plt.plot(x, curve1_y, label="Curve 1", color='blue')
plt.plot(x, curve2_y, label="Curve 2", color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-Intersecting Curves and Their Tangents')
plt.legend()
plt.show()

print("Steering angles for parallel tangents at each pair of points:")
print(np.array(steering_angles[:100]))  

