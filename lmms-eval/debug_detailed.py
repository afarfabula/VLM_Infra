#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from lmms_eval.tasks import TaskManager
import argparse

# Parse arguments like the main CLI does
parser = argparse.ArgumentParser()
parser.add_argument("--tasks", type=str, default=None)
parser.add_argument("--verbosity", type=str, default="INFO")
parser.add_argument("--include_path", type=str, default=None)
parser.add_argument("--model", type=str, default="llava")

args = parser.parse_args(["--tasks", "list"])

# Initialize TaskManager exactly like in cli_evaluate_single
print("Initializing TaskManager...")
task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

print(f"TaskManager initialized.")
print(f"Number of tasks loaded: {len(task_manager.all_tasks)}")

# Check if we have tasks
if len(task_manager.all_tasks) > 0:
    print("First 10 tasks:")
    for i, task in enumerate(sorted(task_manager.all_tasks)[:10]):
        print(f"  {i+1}. {task}")
else:
    print("No tasks found!")
    
# Now simulate what happens in cli_evaluate_single
if args.tasks == "list":
    print("\n--- Simulating cli_evaluate_single list output ---")
    print("Available Tasks:")
    print(" - " + "\n - ".join(sorted(task_manager.all_tasks)))