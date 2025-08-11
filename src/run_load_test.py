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

# Import the load test function from the server module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Patent Search System - Load Test Runner")
    print("=" * 60)
    
    print("\nThis test will simulate up to 100 concurrent users")
    print("Make sure the patent data files are in the current directory")
    
    # Start the Flask server in a subprocess
    print("\nStarting Flask server...")
    server_process = None
    
    try:
        # Start server in background
        server_process = subprocess.Popen(
            [sys.executable, 'patent_backend_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Import and run the load test
        from patent_backend_server import run_load_test
        
        print("\nRunning load tests...")
        results = run_load_test()
        
        print("\n" + "="*60)
        print("LOAD TEST COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - load_test_report.txt: Detailed performance report")
        print("  - load_test_results.png: Performance visualizations")
        
        # Show quick summary
        heavy_load = next((r for r in results if r['total_users'] == 100), None)
        if heavy_load:
            print(f"\n100 Concurrent Users Performance Summary:")
            print(f"  ✓ Throughput: {heavy_load['requests_per_second']:.2f} req/s")
            print(f"  ✓ Avg Response Time: {heavy_load['avg_response_time']:.3f}s")
            print(f"  ✓ P95 Response Time: {heavy_load['p95_response_time']:.3f}s")
            print(f"  ✓ Success Rate: {heavy_load['successful_requests']/heavy_load['total_requests']*100:.1f}%")
            
            if heavy_load['avg_response_time'] < 1.0:
                print("\n🎉 Excellent performance - sub-second average response time!")
            
    except ImportError as e:
        print(f"\nError: Could not import required modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install flask flask-cors aiohttp psutil numpy matplotlib")
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
