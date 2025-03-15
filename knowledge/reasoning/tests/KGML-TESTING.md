# KGML Testing Framework

This framework provides comprehensive testing for the Knowledge Graph Manipulation Language (KGML) and evaluates language models' ability to reason and manipulate knowledge graphs using KGML.

## Overview

The KGML Testing Framework includes:

1. **KGML Executor** - Executes KGML commands against a knowledge graph
2. **Integration Tests** - Verifies end-to-end functionality with actual language models
3. **Test Logger** - Records detailed test information with filesystem persistence
4. **Test Viewer** - Command-line tool to explore test results and analyze performance

![](https://raw.githubusercontent.com/Independent-AI-Labs/local-super-agents/refs/heads/feat/web-search/res/docs/screens/kgml_explorer.png)

## Components

### KGML Executor (`kgml_executor.py`)

The executor is the core component that processes KGML code and applies it to a knowledge graph:

- Processes the KGML AST and executes each command
- Handles all KGML command types (Create, Update, Delete, Evaluate, Navigate)
- Supports control structures like conditionals (IF/ELIF/ELSE) and loops
- Maintains an execution context with variables and results
- Logs detailed execution information

### Integration Tests (`kgml_integration_tests.py`)

The integration tests evaluate how well language models can use KGML to solve reasoning problems:

- Tests basic knowledge graph operations
- Tests model KGML generation capabilities
- Tests multi-step reasoning through iterative KGML execution
- Tests problem-solving at different difficulty levels
- Includes comprehensive evaluation across all test problems

### Test Logger (`kgml_test_logger.py`)

The test logger captures detailed information about each test run:

- Records each request-response pair in separate files
- Organizes test data in a hierarchical directory structure
- Tracks detailed statistics including response times and error rates
- Generates summary reports for test runs
- Provides a consistent interface for logging test events

### Test Viewer (`view_test_results.py`)

The test viewer provides a command-line interface to explore test results:

- List all test runs with summary information
- View detailed statistics for a specific test run
- List all iterations for a specific test
- View request/response pairs and execution results for specific iterations

## Directory Structure

The test results are organized in the following directory structure:

```
kgml_test_logs/
  ├── YYYYMMDD_HHMMSS/         # Run ID (timestamp)
  │   ├── stats.json           # Overall run statistics
  │   ├── summary_report.txt   # Human-readable summary
  │   ├── run.log              # Detailed log file
  │   ├── test_name_1/         # Test directory
  │   │   ├── summary.json     # Test summary
  │   │   ├── iteration_001/   # Iteration directory
  │   │   │   ├── request.kgml        # Request sent to model
  │   │   │   ├── response.kgml       # Response from model
  │   │   │   ├── execution_result.json # Result of executing KGML
  │   │   │   └── metadata.json       # Iteration metadata
  │   │   ├── iteration_002/
  │   │   └── ...
  │   ├── test_name_2/
  │   └── ...
  ├── YYYYMMDD_HHMMSS/         # Another run
  └── ...
```

## Usage

### Running Tests

To run the integration tests:

```bash
pytest knowledge/reasoning/kgml_integration_tests.py -v
```

Or to run a specific test:

```bash
pytest knowledge/reasoning/kgml_integration_tests.py::test_model_kgml_generation -v
```

### Viewing Test Results

# KGML Test Results Viewer

A Gradio-based UI for exploring and analyzing KGML test runs.
## Usage

Run the application:

```bash
python kgml_test_result_ui.py
```

### Options

```bash
python kgml_test_result_ui.py --logs-dir="/path/to/logs" --port=8080 --debug --share
```

- `--logs-dir`: Directory for test logs (default: "kgml_test_logs")
- `--port`: Server port (default: 7860)
- `--debug`: Enable debug mode
- `--share`: Create a shareable link

## Features

- Browse test runs, tests, and iterations
- View request-response pairs
- Analyze execution results
- Visualize test statistics

## Test Problems

The test framework includes problems of varying difficulty:

- **Basic**: Simple reasoning tasks like checking conditions and creating nodes
- **Intermediate**: Multi-step processes with node dependencies and sequencing
- **Advanced**: Complex conditional logic and decision trees

Each problem has:
- A description of the task
- An initial knowledge graph state
- A goal condition that defines success
- A difficulty level

## Statistics Collected

The framework tracks comprehensive statistics:

- **Response Time**: Time taken by the model to generate KGML
- **Syntax Validity**: Whether the generated KGML is syntactically valid
- **Execution Success**: Whether the KGML executed without errors
- **Goal Achievement**: Whether the reasoning task was completed successfully
- **Iterations Required**: Number of steps needed to solve the problem

## Extending the Framework

### Adding New Problems

To add new test problems, update the `problem_definitions` fixture in `kgml_integration_tests.py`:

```python
{
    "id": "new_problem",
    "description": "Description of the problem",
    "initial_kg": "KG►\nKGNODE► Node1 : type=\"Type1\"\n◄",
    "goal_condition": lambda kg: any(node.uid.startswith("Goal") for node in kg.query_nodes()),
    "difficulty": "basic"  # basic, intermediate, or advanced
}
```

### Adding New Test Types

To add new test types, create new test functions in `kgml_integration_tests.py` that:

1. Start a test using the test logger
2. Execute the test logic
3. End the test with success/failure information

```python
def test_new_feature(test_logger):
    test_logger.start_test("new_feature", {"description": "Testing a new feature"})
    
    # Test logic here
    
    test_logger.end_test("new_feature", goal_reached=True, iterations_to_goal=1)
```

## Troubleshooting

- **Missing Test Logs**: Ensure the `kgml_test_logs` directory exists and is writeable
- **Execution Errors**: Check the execution results for detailed error information
- **Model Response Issues**: Examine the raw response.kgml files for unexpected formats