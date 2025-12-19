#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from lmms_eval.tasks import TaskManager

# Initialize TaskManager with debug mode
task_manager = TaskManager(verbosity="DEBUG")

# Print information about loaded tasks
print(f"Number of tasks loaded: {len(task_manager.all_tasks)}")
print("First 10 tasks:")
for i, task in enumerate(sorted(task_manager.all_tasks)[:10]):
    print(f"  {i+1}. {task}")
    
print("\nAll tasks:")
for i, task in enumerate(sorted(task_manager.all_tasks)):
    print(f"  {i+1}. {task}")