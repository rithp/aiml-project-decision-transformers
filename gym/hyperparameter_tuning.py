import os
import subprocess

# --- HARDWARE OPTIMIZATIONS ---
# 1. Force PyTorch to only see the NVIDIA RTX 3050 (usually index 0 for CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2. Enable PyTorch benchmark mode to optimize convolutions for your specific hardware
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

# --- EXPERIMENT SETUP ---
default_K = 20
default_layers = 3
default_heads = 1
default_embd = 128

test_grids = {
    "K": [5, 10, 20, 30, 50],
    "n_layer": [1, 2, 3, 4, 5],
    "n_head": [1, 2, 4, 8, 16],
    "n_embd": [32, 64, 128, 256, 512]
}

env_name = "hopper"
dataset_type = "medium-replay"

os.makedirs("logs", exist_ok=True)
total_runs = sum(len(values) for values in test_grids.values())
print(f"Total experiments to run: {total_runs}")

run_counter = 1

for param_name, param_values in test_grids.items():
    print(f"\n--- Testing sensitivity for: {param_name} ---")
    
    for val in param_values:
        current_K = default_K
        current_layers = default_layers
        current_heads = default_heads
        current_embd = default_embd
        
        if param_name == "K": current_K = val
        elif param_name == "n_layer": current_layers = val
        elif param_name == "n_head": current_heads = val
        elif param_name == "n_embd": current_embd = val
        
        run_name = f"dt_{env_name}_{dataset_type}_{param_name}_{val}"
        log_file = f"logs/{run_name}.txt"
        
        print(f"Run {run_counter}/{total_runs} | Config: K={current_K}, L={current_layers}, H={current_heads}, E={current_embd}")
        
        # --- THE COMMAND ---
        command = [
            "python", "experiment.py",
            "--env", env_name,
            "--dataset", dataset_type,
            "--K", str(current_K),
            "--n_layer", str(current_layers),
            "--n_head", str(current_heads),
            "--n_embd", str(current_embd),
            "--device", "cuda",          # Force the code to use the RTX 3050
            "--batch_size", "64",        # Safe limit for RTX 3050 VRAM
            "--max_iters", "10"          # Limits the total training loop to finish faster. Adjust if needed.
        ]
        
        with open(log_file, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=subprocess.STDOUT)
            
        run_counter += 1

print("\nAll independent sensitivity experiments completed! Check the 'logs' folder.")