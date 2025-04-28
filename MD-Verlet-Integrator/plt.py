import matplotlib.pyplot as plt
import numpy as np

# Load the data from the .dat file
# Replace 'data.dat' with the actual filename
data = np.loadtxt('md_energy.dat')

#print(data)

# Extract time and energy columns
time = data[:, 0]
energy = data[:, 1]

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(time, energy, label='Energy vs Time', color='blue')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Time vs Energy Plot')
plt.legend()
plt.grid(True)
plt.show()
