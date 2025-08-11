from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import asyncio
import aiohttp
from typing import Dict, List, Any
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our search engine
from patent_search_engine import PatentSearchEngine

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.secret_key = 'your-secret-key-here'  # Change this in production

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to root, then into data folder
data_dir = os.path.join(script_dir, '..', 'data', 'patent_data_small')
data_dir = os.path.abspath(data_dir)

# Initialize search engine
print(f"Initializing search engine with data from: {data_dir}")
search_engine = PatentSearchEngine(data_directory=data_dir)

# Performance monitoring
performance_metrics = {
    'total_requests': 0,
    'response_times': [],
    'concurrent_requests': 0,
    'max_concurrent': 0,
    'errors': 0,
    'start_time': time.time()
}

# Thread pool for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=50)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'requests_per_second': [],
            'avg_response_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'concurrent_users': [],
            'timestamp': []
        }
        self.lock = threading.Lock()
    
    def record_metric(self, metric_type: str, value: float):
        with self.lock:
            if metric_type not in self.metrics:
                self.metrics[metric_type] = []
            self.metrics[metric_type].append(value)
            
    def get_current_stats(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'active_threads': threading.active_count()
        }

monitor = PerformanceMonitor()

@app.before_request
def before_request():
    """Track concurrent requests"""
    performance_metrics['concurrent_requests'] += 1
    performance_metrics['max_concurrent'] = max(
        performance_metrics['max_concurrent'],
        performance_metrics['concurrent_requests']
    )

@app.after_request
def after_request(response):
    """Track request completion"""
    performance_metrics['concurrent_requests'] -= 1
    return response

@app.route('/api/search', methods=['POST'])
def search():
    """Perform patent search"""
    start_time = time.time()
    performance_metrics['total_requests'] += 1
    
    try:
        data = request.json
        query = data.get('query', '')
        
        # Perform search
        results = search_engine.search(query, timed=True)
        
        # Track performance
        response_time = time.time() - start_time
        performance_metrics['response_times'].append(response_time)
        monitor.record_metric('response_time', response_time)
        
        # Add performance info to response
        results['performance_info'] = {
            'response_time': response_time,
            'concurrent_requests': performance_metrics['concurrent_requests']
        }
        
        return jsonify(results)
        
    except Exception as e:
        performance_metrics['errors'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current performance statistics"""
    uptime = time.time() - performance_metrics['start_time']
    avg_response_time = np.mean(performance_metrics['response_times'][-100:]) if performance_metrics['response_times'] else 0
    
    stats = {
        'total_requests': performance_metrics['total_requests'],
        'requests_per_second': performance_metrics['total_requests'] / uptime if uptime > 0 else 0,
        'concurrent_requests': performance_metrics['concurrent_requests'],
        'max_concurrent': performance_metrics['max_concurrent'],
        'avg_response_time': avg_response_time,
        'p95_response_time': np.percentile(performance_metrics['response_times'], 95) if performance_metrics['response_times'] else 0,
        'p99_response_time': np.percentile(performance_metrics['response_times'], 99) if performance_metrics['response_times'] else 0,
        'errors': performance_metrics['errors'],
        'error_rate': performance_metrics['errors'] / performance_metrics['total_requests'] if performance_metrics['total_requests'] > 0 else 0,
        'uptime_seconds': uptime,
        'system': monitor.get_current_stats()
    }
    
    return jsonify(stats)

# Concurrency testing functions
async def simulate_user_search(session: aiohttp.ClientSession, user_id: int, num_searches: int = 5):
    """Simulate a single user performing multiple searches"""
    queries = [
        'wheel bearing',
        'abstract: heat shield',
        'class: B60B',
        '{class_prefix: "B60B", abstract: "wheel hub"}',
        'pneumatic tire',
        'claims: monofilament',
        'title: spoke',
        '{class_prefix: "B60C", title_keywords: "tire", abstract: "rubber"}',
        'torque transmission',
        'vehicle wheel assembly'
    ]
    
    results = []
    
    for i in range(num_searches):
        query = random.choice(queries)
        start_time = time.time()
        
        try:
            async with session.post(
                'http://localhost:5000/api/search',
                json={'query': query},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                result = await resp.json()
                response_time = time.time() - start_time
                
                results.append({
                    'user_id': user_id,
                    'query': query,
                    'response_time': response_time,
                    'status': 'success',
                    'result_count': result.get('count', 0),
                    'concurrent_requests': result.get('performance_info', {}).get('concurrent_requests', 0)
                })
        
        except Exception as e:
            results.append({
                'user_id': user_id,
                'query': query,
                'response_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            })
        
        # Random delay between searches
        await asyncio.sleep(random.uniform(0.1, 0.5))
    
    return results

async def load_test(num_users: int = 100, searches_per_user: int = 5):
    """Simulate multiple concurrent users"""
    print(f"\nStarting load test with {num_users} concurrent users...")
    print(f"Each user will perform {searches_per_user} searches")
    print("-" * 60)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_users):
            task = simulate_user_search(session, i, searches_per_user)
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
    
    results = [item for sublist in all_results for item in sublist]
    
    total_time = time.time() - start_time
    successful_requests = [r for r in results if r['status'] == 'success']
    failed_requests = [r for r in results if r['status'] == 'error']
    response_times = [r['response_time'] for r in successful_requests]
    
    max_concurrent = max([r.get('concurrent_requests', 0) for r in successful_requests] + [0])
    
    stats = {
        'total_users': num_users,
        'searches_per_user': searches_per_user,
        'total_requests': len(results),
        'successful_requests': len(successful_requests),
        'failed_requests': len(failed_requests),
        'total_time': total_time,
        'requests_per_second': len(results) / total_time,
        'avg_response_time': np.mean(response_times) if response_times else 0,
        'min_response_time': np.min(response_times) if response_times else 0,
        'max_response_time': np.max(response_times) if response_times else 0,
        'p50_response_time': np.percentile(response_times, 50) if response_times else 0,
        'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
        'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
        'max_concurrent_observed': max_concurrent
    }
    
    return stats, results

def generate_performance_report(stats_list: List[Dict]):
    """Generate a detailed performance report with visualizations"""
    
    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    results_dir = os.path.abspath(results_dir)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Patent Search System - Load Test Results', fontsize=16)
    
    # Extract data
    scenarios = [s['scenario'] for s in stats_list]
    users = [s['total_users'] for s in stats_list]
    rps = [s['requests_per_second'] for s in stats_list]
    p50_times = [s['p50_response_time'] for s in stats_list]
    p95_times = [s['p95_response_time'] for s in stats_list]
    p99_times = [s['p99_response_time'] for s in stats_list]
    success_rates = [(s['successful_requests']/s['total_requests']*100) for s in stats_list]
    cpu_usage = [s['cpu_during_load'] for s in stats_list]
    memory_usage = [s['memory_during_load'] for s in stats_list]
    
    # 1. Throughput vs Load
    axes[0, 0].bar(range(len(scenarios)), rps, color='steelblue')
    axes[0, 0].set_xticks(range(len(scenarios)))
    axes[0, 0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Requests/Second')
    axes[0, 0].set_title('Throughput vs Load')
    axes[0, 0].grid(True, alpha=0.3)
    
    for i, v in enumerate(rps):
        axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center')
    
    # 2. Response Time Percentiles
    x = np.arange(len(scenarios))
    width = 0.2
    
    axes[0, 1].bar(x - width, p50_times, width, label='P50 (Median)', color='green')
    axes[0, 1].bar(x, p95_times, width, label='P95', color='orange')
    axes[0, 1].bar(x + width, p99_times, width, label='P99', color='red')
    
    axes[0, 1].set_ylabel('Response Time (seconds)')
    axes[0, 1].set_title('Response Time Percentiles')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success Rate
    colors = ['green' if sr >= 99 else 'orange' if sr >= 95 else 'red' for sr in success_rates]
    
    axes[1, 0].bar(range(len(scenarios)), success_rates, color=colors)
    axes[1, 0].set_xticks(range(len(scenarios)))
    axes[1, 0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Request Success Rate')
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].axhline(y=99, color='green', linestyle='--', alpha=0.5, label='Target (99%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i, v in enumerate(success_rates):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 4. Resource Utilization
    x = np.arange(len(scenarios))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, cpu_usage, width, label='CPU %', color='blue')
    axes[1, 1].bar(x + width/2, memory_usage, width, label='Memory %', color='green')
    
    axes[1, 1].set_ylabel('Usage (%)')
    axes[1, 1].set_title('Resource Utilization During Load')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with timestamp
    plot_filename = f'load_test_results_{timestamp}.png'
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    heavy_load = next((s for s in stats_list if s['total_users'] == 100), stats_list[-1])
    
    report = f"""Patent Search System - Performance Test Report
=============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Peak Performance Metrics (100 concurrent users):
• Throughput: {heavy_load['requests_per_second']:.2f} requests/second
• Average Response Time: {heavy_load['avg_response_time']:.3f} seconds
• 95th Percentile Response: {heavy_load['p95_response_time']:.3f} seconds
• 99th Percentile Response: {heavy_load['p99_response_time']:.3f} seconds
• Success Rate: {heavy_load['successful_requests']/heavy_load['total_requests']*100:.2f}%
• Max Concurrent Observed: {heavy_load['max_concurrent_observed']}

Test Scenarios:
"""
    
    # Add details for each scenario
    for stats in stats_list:
        report += f"""
{stats['scenario']}:
  Users: {stats['total_users']} concurrent
  Total Requests: {stats['total_requests']}
  Throughput: {stats['requests_per_second']:.2f} req/s
  Avg Response Time: {stats['avg_response_time']:.3f}s
  Success Rate: {stats['successful_requests']/stats['total_requests']*100:.1f}%
"""
    
    report += f"\nDetailed visualizations saved in: {plot_filename}"
    
    # Save report with timestamp
    report_filename = f'load_test_report_{timestamp}.txt'
    report_path = os.path.join(results_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Also save a JSON file with raw data for later analysis
    json_filename = f'load_test_data_{timestamp}.json'
    json_path = os.path.join(results_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'scenarios': stats_list,
            'summary': {
                'peak_throughput': max(s['requests_per_second'] for s in stats_list),
                'best_avg_response_time': min(s['avg_response_time'] for s in stats_list),
                'max_concurrent_observed': max(s['max_concurrent_observed'] for s in stats_list)
            }
        }, f, indent=2)
    
    print("\nLoad test complete!")
    print(f"Results saved to: {results_dir}")
    print("Generated files:")
    print(f"  - {report_filename}")
    print(f"  - {plot_filename}")
    print(f"  - {json_filename}")

def run_load_test():
    """Run the load test and generate report"""
    print("\nPATENT SEARCH SYSTEM - LOAD TEST")
    print("="*60)
    
    initial_stats = monitor.get_current_stats()
    print(f"Initial System Status:")
    print(f"  CPU: {initial_stats['cpu_percent']:.1f}%")
    print(f"  Memory: {initial_stats['memory_percent']:.1f}%")
    
    scenarios = [
        (10, 5, "Light Load"),
        (50, 5, "Medium Load"),
        (100, 5, "Heavy Load"),
        (100, 10, "Extended Heavy Load")
    ]
    
    all_stats = []
    
    for num_users, searches_per_user, scenario_name in scenarios:
        print(f"\n{scenario_name}: {num_users} users, {searches_per_user} searches each")
        print("-" * 40)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats, results = loop.run_until_complete(load_test(num_users, searches_per_user))
        loop.close()
        
        load_stats = monitor.get_current_stats()
        
        stats['cpu_during_load'] = load_stats['cpu_percent']
        stats['memory_during_load'] = load_stats['memory_percent']
        stats['scenario'] = scenario_name
        
        all_stats.append(stats)
        
        print(f"Results:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Successful: {stats['successful_requests']} ({stats['successful_requests']/stats['total_requests']*100:.1f}%)")
        print(f"  Failed: {stats['failed_requests']}")
        print(f"  Throughput: {stats['requests_per_second']:.2f} req/s")
        print(f"  Avg response time: {stats['avg_response_time']:.3f}s")
    
    generate_performance_report(all_stats)
    return all_stats  # Return the stats so run_load_test.py can use them

# Start the Flask server
if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)