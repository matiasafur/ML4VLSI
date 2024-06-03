####################################################################################################
# Network layer computation

# Define parameters for each layer
layers = {
    "Conv2d_1_1": {"output_shape": (64, 32, 32), "kernel_size": (3, 3), "params": 1792},
    "Conv2d_1_3": {"output_shape": (128, 16, 16), "kernel_size": (3, 3), "params": 73856},
    "Conv2d_1_5": {"output_shape": (256, 8, 8), "kernel_size": (3, 3), "params": 295168},
    "Linear_1_7": {"output_shape": (1024,), "input_size": 256 * 4 * 4, "params": 4195328},
    "Linear_1_9": {"output_shape": (512,), "input_size": 1024, "params": 524800},
    "Linear_1_10": {"output_shape": (10,), "input_size": 512, "params": 5130}
}

# Calculate the number of multiplications and additions for each layer
total_multiplications = 0
total_additions = 0

# Conv2d Layers
for key, value in layers.items():
    if "Conv2d" in key:
        out_channels, out_height, out_width = value["output_shape"]
        kernel_height, kernel_width = value["kernel_size"]
        num_multiplications = out_channels * out_height * out_width * kernel_height * kernel_width
        num_additions = num_multiplications  # Same as number of multiplications
        total_multiplications += num_multiplications
        total_additions += num_additions

# Linear Layers
for key, value in layers.items():
    if "Linear" in key:
        output_size = value["output_shape"][0]
        input_size = value["input_size"]
        num_multiplications = output_size * input_size
        num_additions = num_multiplications  # Same as number of multiplications
        total_multiplications += num_multiplications
        total_additions += num_additions

####################################################################################################
# Mac requirements estimation

# Area and power specs from the previous calculation
area_per_mac = 2606.4  # in um^2
dynamic_power_per_mac = 2.6408  # in uW/MHz
leakage_power_per_mac = 18.5  # in nW
clock_frequency = 100  # in MHz

# Estimation calculations
mac_units = 1  # start with 1 MAC
cycles_per_operation = 1  # each MAC operation takes 1 clock cycle

# Total operations
total_operations = total_multiplications + total_additions

# Calculate the number of clock cycles needed
clock_cycles_needed = total_operations / mac_units

# Calculate the total area
total_area = mac_units * area_per_mac

# Calculate the total power consumption
total_dynamic_power = mac_units * dynamic_power_per_mac * clock_frequency
total_leakage_power = mac_units * leakage_power_per_mac
total_power = total_dynamic_power + total_leakage_power

# Results
print(f"Number of MACs: {mac_units}")
print(f"Clock cycles needed: {clock_cycles_needed:.2f}")
print(f"Total area (um^2): {total_area:.2f}")
print(f"Total power consumption (uW): {total_power:.2f}")

####################################################################################################
# Available area MAC estimation

# Maximum available area in um^2
max_area_available = 1.5 * 1e6  # 1.5 mm^2 in um^2

# Calculate the number of MACs that can be fabricated
max_mac_units = max_area_available // area_per_mac

# Calculate the number of clock cycles needed with max MACs
clock_cycles_needed_max_macs = total_operations / max_mac_units

# Recalculate the total power consumption with max MACs
total_dynamic_power_max_macs = max_mac_units * dynamic_power_per_mac * clock_frequency
total_leakage_power_max_macs = max_mac_units * leakage_power_per_mac
total_power_max_macs = total_dynamic_power_max_macs + total_leakage_power_max_macs

# Print the results
print(f"Maximum number of MACs that can be fabricated: {max_mac_units}")
print(f"Clock cycles needed with max MACs: {clock_cycles_needed_max_macs:.2f}")
print(f"Total power consumption with max MACs (uW): {total_power_max_macs:.2f}")
