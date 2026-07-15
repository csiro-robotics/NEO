import os
import shlex
import subprocess
import torch

from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

def check_available_gpus():
    # Check the number of available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {available_gpus}")
    return available_gpus

def run_script_on_gpu(gpu_id, command_queue):
    while not command_queue.empty():
        # Get the next command from the queue
        command = command_queue.get()
        print(f"Running: {command} on GPU {gpu_id}")
        # Set the GPU and run the training script
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        subprocess.run(shlex.split(command), env=env, check=True)
        print(f"Finished: {command} on GPU {gpu_id}")

def run_on_multiple_gpus(commands: List[str]):
    available_gpus = 1 # check_available_gpus()

    # Create a queue and add all commands to it
    command_queue = Queue()
    for command in commands:
        command_queue.put(command)

    # Use ThreadPoolExecutor to manage GPU workers
    with ThreadPoolExecutor(max_workers=available_gpus) as executor:
        futures = []
        # Assign one GPU per worker
        for gpu_id in range(available_gpus):
            futures.append(executor.submit(run_script_on_gpu, gpu_id, command_queue))

        # Wait for all futures to complete (i.e., all tasks to finish)
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":

    # Example commands to run on multiple GPUs
    commands = [
        # "python3 scripts/run_removal.py --load-config orange.yml",
        # "python3 scripts/run_removal.py --load-config red_robot.yml",
        # "python3 scripts/run_removal.py --load-config mustard_bottle.yml",
        # "python3 scripts/run_removal.py --load-config tomato_soup_can.yml",
        # "python3 scripts/run_removal.py --load-config spam.yml"
        # "python3 scripts/run_reconfiguration.py --load-config sugar.yml",
        # "python3 scripts/run_reconfiguration.py --load-config orange.yml",
        # "python3 scripts/run_reconfiguration.py --load-config rubiks_cube.yml",
        # "python3 scripts/run_reconfiguration.py --load-config spam.yml",
        # "python3 scripts/run_reconfiguration.py --load-config banana.yml",
        # "python3 scripts/run_reconfiguration.py --load-config tomato_soup_can.yml",
        "python3 scripts/run_assembly.py --load-config assembly_0.yml",
        "python3 scripts/run_assembly.py --load-config assembly_1.yml",
        "python3 scripts/run_assembly.py --load-config assembly_2.yml",
        "python3 scripts/run_assembly.py --load-config assembly_3.yml",
        "ns-render dataset --load-config outputs/real/assembly/single/assembly/lego_bricks_1/before/neo/2025-07-18_113809/replace-nerf/config.yml --output-path renders_assembly/lego_bricks_1 --rendered-output-names rgb --image-format png --split train+test",
        "ns-render dataset --load-config outputs/real/assembly/single/assembly/red_cup/before/neo/2025-07-18_113809/replace-nerf/config.yml --output-path renders_assembly/red_cup --rendered-output-names rgb --image-format png --split train+test",
        "ns-render dataset --load-config outputs/real/assembly/single/assembly/lego_bricks_2/before/neo/2025-07-18_113809/replace-nerf/config.yml --output-path renders_assembly/lego_bricks_2 --rendered-output-names rgb --image-format png --split train+test",
        "ns-render dataset --load-config outputs/real/assembly/single/assembly/golf_ball/before/neo/2025-07-18_113809/replace-nerf/config.yml --output-path renders_assembly/golf_ball --rendered-output-names rgb --image-format png --split train+test",
    ]

    run_on_multiple_gpus(commands)
