# Mac area calculations

# Define constants for the dimensions and power of each type of gate
gate_specs = {
    "AND2": {"height": 1.8, "width": 1.0, "dynamic_power": 0.0021, "leakage_power": 14.23},
    "NAND2": {"height": 1.8, "width": 0.8, "dynamic_power": 0.00095, "leakage_power": 8.2},
    "INV": {"height": 1.8, "width": 0.6, "dynamic_power": 0.00074, "leakage_power": 5.14}
}

# Define the number of gates used in the multiplier and the accumulator
num_gates_multiplier = 1000  # for AND2 gates in the multiplier
num_gates_accumulator_nand2 = 320  # for NAND2 gates in the accumulator
num_gates_accumulator_inv = 320  # for INV gates in the accumulator

# Calculate the area for each component
area_multiplier = num_gates_multiplier * gate_specs["AND2"]["height"] * gate_specs["AND2"]["width"]
area_accumulator_nand2 = num_gates_accumulator_nand2 * gate_specs["NAND2"]["height"] * gate_specs["NAND2"]["width"]
area_accumulator_inv = num_gates_accumulator_inv * gate_specs["INV"]["height"] * gate_specs["INV"]["width"]
total_area = area_multiplier + area_accumulator_nand2 + area_accumulator_inv

# Calculate the dynamic power consumption for each component
dynamic_power_multiplier = num_gates_multiplier * gate_specs["AND2"]["dynamic_power"]
dynamic_power_accumulator_nand2 = num_gates_accumulator_nand2 * gate_specs["NAND2"]["dynamic_power"]
dynamic_power_accumulator_inv = num_gates_accumulator_inv * gate_specs["INV"]["dynamic_power"]
total_dynamic_power = dynamic_power_multiplier + dynamic_power_accumulator_nand2 + dynamic_power_accumulator_inv

# Calculate the leakage power consumption for each component
leakage_power_multiplier = num_gates_multiplier * gate_specs["AND2"]["leakage_power"]
leakage_power_accumulator_nand2 = num_gates_accumulator_nand2 * gate_specs["NAND2"]["leakage_power"]
leakage_power_accumulator_inv = num_gates_accumulator_inv * gate_specs["INV"]["leakage_power"]
total_leakage_power = leakage_power_multiplier + leakage_power_accumulator_nand2 + leakage_power_accumulator_inv

# Print the results
print(f"Total Area (µm²): {total_area:.2f}")
print(f"Total Dynamic Power (µW/MHz): {total_dynamic_power:.4f}")
print(f"Total Leakage Power (nW): {total_leakage_power:.2f}")
