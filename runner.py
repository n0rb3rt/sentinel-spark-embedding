#!/usr/bin/env python3
"""
Generic job runner - executes any job from the sentinel-processing package
"""
import sys
import importlib

def main():
    if len(sys.argv) < 2:
        print("Usage: python runner.py <job_name> [args...]")
        sys.exit(1)
    
    job_name = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove runner.py from args
    
    # Import and run the job
    module = importlib.import_module(f'sentinel_processing.jobs.{job_name}')
    module.main()

if __name__ == "__main__":
    main()