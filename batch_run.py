import subprocess
import sys
print(sys.executable)
#python_exec = sys.executable  # Uses the *current* Python interpreter

# Define your combinations
configs = [
    # RMSprop runs
    ("relu", "adamax"),
    ("leaky_relu", "adamax"),
    ("swish", "adamax"),
    ("mish", "adamax"),
    ("sigmoid", "adamax"),
    ("reverse_relu", "adamax"),
    ("reverse_leaky_relu", "adamax"),
     
    ("oscillating_relu", "adamax"),
    ("staircase_relu", "adamax"),
    ("sin_relu", "adamax"),
    ("bent_identity", "adamax"),
    ("elu_sin", "adamax"),
    ("chaotic_relu", "adamax"),
    ("gravity", "adamax"),
    ("gravity_x", "adamax"),
    ("gravity_x_swish", "adamax"),
    ("sin_exp_decay", "adamax"),

]

python_exec = sys.executable

for activation, optimizer in configs:
    print(f"\n[🚀] Running with activation={activation}, optimizer={optimizer}")
    command = [
        python_exec,
        "train.py",
        "--activation", activation,
        "--optimizer-type", optimizer
    ]
    subprocess.run(command)