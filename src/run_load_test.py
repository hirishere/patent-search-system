#!/usr/bin/env python3
"""
Load Test Runner for Patent Search System
Run this script to simulate concurrent users and generate performance reports
"""

import sys
import time
import os
import subprocess
import signal
import requests

# Add the src directory to Python path since backend is in src/
script_dir = os.path.dirname(os.path.abspath(__file__))

def wait_for_server(max_attempts=10):
    """Wait for the server to be ready"""
    print("Checking if server is ready...")
    for i in range(max_attempts):
        try:
            response = requests.get('http://localhost:5000/api/stats')
            if response.status_code == 200:
                print("✓ Server is ready!")
                return True
        except:
            pass
        print(f"  Waiting... ({i+1}/{max_attempts})")
        time.sleep(1)
    return False

def main():
    print("Patent Search System - Load Test Runner")
    print("=" * 60)
    
    print("\nThis test will simulate up to 100 concurrent users")
    print("Make sure the patent data files are in data/patent_data_small/")
    
    # Start the Flask server in a subprocess
    print("\nStarting Flask server...")
    server_process = None
    
    try:
        # Start server in background - it's in the src directory
        server_path = os.path.join(script_dir, 'patent_backend_server.py')
        server_process = subprocess.Popen(
            [sys.executable, server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        if not wait_for_server():
            print("ERROR: Server failed to start!")
            # Print server output for debugging
            stdout, stderr = server_process.communicate(timeout=1)
            if stdout:
                print("Server stdout:", stdout.decode())
            if stderr:
                print("Server stderr:", stderr.decode())
            sys.exit(1)
        
        # Import and run the load test
        from patent_backend_server import run_load_test
        
        print("\nRunning load tests...")
        results = run_load_test()
        
        print("\n" + "="*60)
        print("LOAD TEST COMPLETE!")
        print("="*60)
        
        # Show where files were saved
        results_dir = os.path.join(script_dir, 'results')
        print(f"\nResults saved to: {results_dir}")
        print("Generated files with timestamps - check the results directory")
        
        # Show quick summary
        heavy_load = next((r for r in results if r['total_users'] == 100), None)
        if heavy_load:
            print(f"\n100 Concurrent Users Performance Summary:")
            print(f"  ✓ Throughput: {heavy_load['requests_per_second']:.2f} req/s")
            print(f"  ✓ Avg Response Time: {heavy_load['avg_response_time']:.3f}s")
            print(f"  ✓ P95 Response Time: {heavy_load['p95_response_time']:.3f}s")
            print(f"  ✓ Success Rate: {heavy_load['successful_requests']/heavy_load['total_requests']*100:.1f}%")
            
            if heavy_load['avg_response_time'] < 1.0 and heavy_load['avg_response_time'] > 0:
                print("\nExcellent performance - sub-second average response time!")
            
    except ImportError as e:
        print(f"\nError: Could not import required modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install flask flask-cors aiohttp psutil numpy matplotlib requests")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    except Exception as e:
        print(f"\nError during load test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up server process
        if server_process:
            print("\nStopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()