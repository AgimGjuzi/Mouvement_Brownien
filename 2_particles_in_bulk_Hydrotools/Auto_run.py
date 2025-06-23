import os
import shutil
import tempfile
import subprocess

# ---- SETTINGS ----
r_values = [8]
num_runs = 1
use_traps = True  # ← Change to False if you don’t want traps
base_input_file = "inputfile_blobs.dat"
output_base_dir = "outputs"

# ---- TRAP PARAMETERS (used only if use_traps = True) ----
trap_centers_template = "trap_centers  [[0, 0, 0], [{}, 0, 0]]"
trap_mode = "trap_mode  individual"
trap_stiffness = "trap_stiffness  [[0.1, 0.1, 0.01], [0.1, 0.1, 0.01]]"

# ---- MAIN LOOP ----
os.makedirs(output_base_dir, exist_ok=True)

for r in r_values:
    r_str = f"{r:.2f}"  # ← Format r with 2 decimal places
    for run in range(1, num_runs + 1):
        tag = f"r_{r_str}_run{run}_blobs"
        run_dir = os.path.join(output_base_dir, f"r_{r_str}", f"run{run}")
        os.makedirs(run_dir, exist_ok=True)

        # Read base input file
        with open(base_input_file, "r") as f:
            content = f.read()

        # Add or remove trap lines
        if use_traps:
            content += f"\n{trap_centers_template.format(r)}\n{trap_mode}\n{trap_stiffness}\n"
        else:
            for key in ["trap_centers", "trap_mode", "trap_stiffness"]:
                content = "\n".join(line for line in content.splitlines() if key not in line)

        # Set output name
        content += f"\noutput_name = {tag}\n"

        # Save temp input file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".dat", delete=False) as temp_file:
            temp_file.write(content)
            temp_input_path = temp_file.name

        # Run simulations
        print(f"▶️ Running: {tag} (traps: {'yes' if use_traps else 'no'})")
        subprocess.run(["python", "create_initial_distrib.py", str(r)], check=True)
        subprocess.run(["python", "multi_bodies.py", "--input-file", temp_input_path], check=True)

        # Move output files matching the tag
        for f in os.listdir():
            if tag in f and os.path.isfile(f):
                shutil.move(f, os.path.join(run_dir, f))

        print(f"✅ Done: {tag}")


print("\n🏁 All done.")
