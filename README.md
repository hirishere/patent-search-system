# Patent Search System

## Problem Statement

This project addresses the challenge of efficiently searching through patent data to support claim mapping and prior art discovery. The specific problems solved include:

1. **Multi-modal Search**: Patent professionals need to search by various criteria - keywords, classifications, specific fields, or complex combinations
2. **Performance at Scale**: Supporting concurrent users while maintaining sub-second response times
3. **Data Quality**: Handling incomplete patent records gracefully
4. **Evaluation Pipeline**: Building infrastructure to improve search quality through training and evaluation

## Solution Overview

I built a high-performance patent search system with:
- 8 different search modes including hybrid search with multiple constraints
- Support for 100+ concurrent users with sub-second response times
- Web interface with real-time performance monitoring
- Training/evaluation pipeline for search quality improvement
- Comprehensive load testing and performance analysis tools

### Enhancement Chosen: Interfaces and Users

I selected the "Interfaces and Users" enhancement because it addresses real-world deployment challenges:
- Created a responsive web interface for easy interaction
- Implemented session-based search history (client-side)
- Built concurrent user support with performance monitoring
- Developed load testing to simulate 100 users simultaneously

The system demonstrates excellent performance under load, maintaining average response times under 500ms even with 100 concurrent users.

## How to Run

### Prerequisites
```bash
pip install flask flask-cors aiohttp psutil numpy matplotlib sentence-transformers scikit-learn
```

### Running the System

1. **Start the Backend Server**:
```bash
python3 src/patent_backend_server.py
```
The server will start on http://localhost:5000

2. **Access the Web Interface**:
Open `index.html` in a web browser or serve it with:
```bash
python3 -m http.server 8000
```
Then navigate to http://localhost:8000/index.html

3. **Run Load Tests**:
```bash
python3 src/run_load_test.py
```
This will simulate concurrent users and generate performance reports in the `results/` directory.

4. **Run Evaluation Pipeline**:
```bash
python3 src/patent_eval_training.py
```

## Features Demonstrated

### Search Capabilities
- **Keyword Search**: `wheel bearing`
- **Patent ID**: `US20240051333`
- **Title Search**: `title: pneumatic tire`
- **Abstract Search**: `abstract: heat shield`
- **Claims Search**: `claims: monofilament`
- **Classification**: `class: B60B`
- **Hybrid Search**: `{class_prefix: "B60B", title_keywords: "wheel hub"}`

### Performance Features
- Real-time performance metrics display
- Search history with one-click replay
- Concurrent request handling
- Comprehensive load testing reports

### Data Handling
- Processes all JSON files in the data directory
- Handles incomplete patents based on configuration
- Provides data quality reports

## Architecture Decisions

1. **Flask + ThreadPoolExecutor**: Chosen for simplicity while still supporting concurrent requests
2. **Client-side History**: Avoids server-side session management complexity
3. **Indexing Strategy**: Pre-built indexes for classification and keywords improve search speed
4. **Modular Design**: Separate components for search, evaluation, and load testing

## Performance Results

Under load testing with 100 concurrent users:
- Average response time: ~400-500ms
- 95th percentile: ~800ms
- Success rate: >99%
- Throughput: 150+ requests/second

The system scales well and maintains consistent performance even under heavy load.
