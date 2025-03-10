#!/usr/bin/env python3
"""
Script to view and analyze test results from KGML tests.
This script provides a command-line interface to explore test
logs, statistics, and individual request-response pairs.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def list_test_runs(base_dir: str = "kgml_test_logs") -> List[Dict[str, Any]]:
    """
    List all test runs in the specified directory.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"No test logs found in {base_dir}")
        return []

    runs = []
    for run_dir in sorted(base_path.iterdir(), key=lambda p: p.name, reverse=True):
        if not run_dir.is_dir():
            continue

        stats_file = run_dir / "stats.json"
        if not stats_file.exists():
            continue

        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)

                # Calculate success rate safely
                total_responses = max(1, stats.get("total_responses", 1))
                valid_responses = stats.get("valid_responses", 0)
                success_rate = (valid_responses / total_responses) * 100

                # Extract key information
                run_info = {
                    "run_id": stats.get("run_id", run_dir.name),
                    "start_time": stats.get("start_time", "Unknown"),
                    "end_time": stats.get("end_time", "Unknown"),
                    "total_tests": stats.get("total_tests", 0),
                    "success_rate": f"{success_rate:.1f}%",
                    "path": str(run_dir)
                }
                runs.append(run_info)
        except Exception as e:
            print(f"Error reading stats for {run_dir}: {e}")

    return runs


def show_run_summary(run_path: str) -> None:
    """
    Show a summary of a specific test run.
    """
    run_dir = Path(run_path)
    if not run_dir.exists():
        print(f"Test run directory not found: {run_path}")
        return

    stats_file = run_dir / "stats.json"
    if not stats_file.exists():
        print(f"Stats file not found for run: {run_path}")
        return

    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

        print("\n" + "=" * 50)
        print(f"Test Run: {stats.get('run_id', 'Unknown')}")
        print("=" * 50)

        # Parse start and end times safely
        start_time_str = stats.get("start_time")
        start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()

        end_time_str = stats.get("end_time")
        end_time = datetime.fromisoformat(end_time_str) if end_time_str else None

        duration = end_time - start_time if end_time else "In progress"

        print(f"Started: {start_time}")
        print(f"Ended: {end_time or 'In progress'}")
        print(f"Duration: {duration}")
        print(f"Total Tests: {stats.get('total_tests', 0)}")
        print(f"Total Prompts: {stats.get('total_prompts', 0)}")

        # Calculate percentages safely
        total_responses = max(1, stats.get('total_responses', 1))
        valid_responses = stats.get('valid_responses', 0)
        syntax_errors = stats.get('syntax_errors', 0)
        processing_errors = stats.get('processing_errors', 0)

        valid_pct = (valid_responses / total_responses) * 100
        syntax_pct = (syntax_errors / total_responses) * 100
        exec_pct = (processing_errors / total_responses) * 100

        print(f"Valid Responses: {valid_responses} / {stats.get('total_responses', 0)} ({valid_pct:.1f}%)")
        print(f"Syntax Errors: {syntax_errors} / {stats.get('total_responses', 0)} ({syntax_pct:.1f}%)")
        print(f"Execution Errors: {processing_errors} / {stats.get('total_responses', 0)} ({exec_pct:.1f}%)")

        # Display response time statistics safely
        avg_response_time = stats.get("avg_response_time")
        min_response_time = stats.get("min_response_time")
        max_response_time = stats.get("max_response_time")

        if avg_response_time is not None:
            print(f"\nResponse Time Statistics:")
            print(f"Average: {avg_response_time:.2f}s")
            if min_response_time is not None:
                print(f"Minimum: {min_response_time:.2f}s")
            if max_response_time is not None:
                print(f"Maximum: {max_response_time:.2f}s")

        # List individual tests
        print("\nIndividual Tests:")
        print("-" * 50)
        for test_name, test_stats in stats.get("tests", {}).items():
            goal_reached = test_stats.get("goal_reached")
            status = "✅ PASSED" if goal_reached else "❌ FAILED" if goal_reached is not None else "⚠️ UNKNOWN"
            print(f"{test_name}: {status}")
            print(f"  Iterations: {test_stats.get('iterations', 0)}")
            print(f"  Valid Responses: {test_stats.get('valid_responses', 0)} / {test_stats.get('responses', 0)}")

            test_avg_response_time = test_stats.get("avg_response_time")
            if test_avg_response_time is not None:
                print(f"  Avg Response Time: {test_avg_response_time:.2f}s")
            print()

        # Check for comprehensive summary
        summary_file = run_dir / "comprehensive_summary.txt"
        if summary_file.exists():
            print("\nComprehensive Evaluation Summary:")
            print("-" * 50)
            with open(summary_file, "r", encoding="utf-8") as f:
                print(f.read())

    except Exception as e:
        print(f"Error reading run summary: {e}")


def list_test_iterations(run_path: str, test_name: str) -> List[Dict[str, Any]]:
    """
    List all iterations for a specific test in a run.
    """
    run_dir = Path(run_path)

    # Find the test directory (which might be a sanitized version of the name)
    test_dir = None
    for d in run_dir.iterdir():
        if d.is_dir() and (d.name == test_name or test_name.lower() in d.name.lower()):
            test_dir = d
            break

    if not test_dir:
        print(f"Test '{test_name}' not found in run: {run_path}")
        return []

    iterations = []
    for iter_dir in sorted(test_dir.iterdir(), key=lambda p: p.name):
        if not iter_dir.is_dir() or not iter_dir.name.startswith("iteration_"):
            continue

        meta_file = iter_dir / "metadata.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

                # Check for processing result
                has_processing_result = (iter_dir / "processing_result.json").exists()

                # Format response time safely
                response_time = metadata.get('response_time_seconds', 0)
                response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"

                # Extract key information
                iter_info = {
                    "iteration": metadata.get("iteration", iter_dir.name),
                    "timestamp": metadata.get("timestamp", "Unknown"),
                    "response_time": response_time_str,
                    "is_valid": metadata.get("is_valid", False),
                    "has_syntax_errors": metadata.get("has_syntax_errors", False),
                    "processing_success": metadata.get("processing_success", None),
                    "path": str(iter_dir)
                }
                iterations.append(iter_info)
        except Exception as e:
            print(f"Error reading metadata for {iter_dir}: {e}")

    return iterations


def show_iteration_details(iteration_path: str) -> None:
    """
    Show details of a specific test iteration.
    """
    iter_dir = Path(iteration_path)
    if not iter_dir.exists():
        print(f"Iteration directory not found: {iteration_path}")
        return

    # Load metadata
    meta_file = iter_dir / "metadata.json"
    if not meta_file.exists():
        print(f"Metadata file not found for iteration: {iteration_path}")
        return

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("\n" + "=" * 50)
        print(f"Iteration {metadata.get('iteration', 'Unknown')}")
        print("=" * 50)

        print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")

        # Format response time safely
        response_time = metadata.get('response_time_seconds')
        if response_time is not None:
            print(f"Response Time: {response_time:.2f}s")
        else:
            print(f"Response Time: N/A")

        print(f"Valid KGML: {'✅ Yes' if metadata.get('is_valid', False) else '❌ No'}")
        print(f"Syntax Errors: {'❌ Yes' if metadata.get('has_syntax_errors', False) else '✅ No'}")

        # Handle processing success status safely
        processing_success = metadata.get('processing_success')
        if processing_success is not None:
            print(f"Processing Success: {'✅ Yes' if processing_success else '❌ No'}")
        else:
            print(f"Processing Success: ⚠️ Unknown")

        # Show request
        request_file = iter_dir / "request.kgml"
        if request_file.exists():
            print("\nRequest:")
            print("-" * 50)
            with open(request_file, "r", encoding="utf-8") as f:
                print(f.read())

        # Show response
        response_file = iter_dir / "response.kgml"
        if response_file.exists():
            print("\nResponse:")
            print("-" * 50)
            with open(response_file, "r", encoding="utf-8") as f:
                print(f.read())

        # Show processing result
        result_file = iter_dir / "processing_result.json"
        if result_file.exists():
            print("\nProcessing Result:")
            print("-" * 50)
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
                # Format the output for better readability
                if not result.get("success", False):
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"✅ Processing succeeded")
                    print(f"Commands executed: {len(result.get('processing_log', []))}")
                    print(f"Variables set: {len(result.get('variables', {}))}")
                    print(f"Results stored: {len(result.get('results', {}))}")

                    # Show processing log summary
                    if result.get("processing_log"):
                        print("\nProcessing Log Summary:")
                        for idx, entry in enumerate(result["processing_log"], 1):
                            cmd_type = entry.get("command_type", "Unknown")
                            success = "✅" if entry.get("success", False) else "❌"
                            details = entry.get("details", {})
                            entity = ""
                            if isinstance(details, dict):
                                if "entity_type" in details and "uid" in details:
                                    entity = f"{details['entity_type']} {details['uid']}"
                                else:
                                    entity = str(details)
                            else:
                                entity = str(details)
                            print(f"  {idx}. {success} {cmd_type}: {entity}")

    except Exception as e:
        print(f"Error reading iteration details: {e}")


def main():
    """
    Main function for the CLI.
    """
    parser = argparse.ArgumentParser(description="View and analyze KGML test results")
    parser.add_argument("--dir", default="kgml_test_logs", help="Base directory for test logs")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List runs command
    list_runs_parser = subparsers.add_parser("list-runs", help="List all test runs")

    # Show run command
    show_run_parser = subparsers.add_parser("show-run", help="Show details of a specific test run")
    show_run_parser.add_argument("run_id", help="ID of the test run to show")

    # List test iterations command
    list_iterations_parser = subparsers.add_parser("list-iterations", help="List all iterations for a specific test")
    list_iterations_parser.add_argument("run_id", help="ID of the test run")
    list_iterations_parser.add_argument("test_name", help="Name of the test")

    # Show iteration command
    show_iteration_parser = subparsers.add_parser("show-iteration", help="Show details of a specific test iteration")
    show_iteration_parser.add_argument("run_id", help="ID of the test run")
    show_iteration_parser.add_argument("test_name", help="Name of the test")
    show_iteration_parser.add_argument("iteration", help="Iteration number")

    args = parser.parse_args()

    base_dir = args.dir

    if args.command == "list-runs" or not args.command:
        # List all test runs
        runs = list_test_runs(base_dir)
        if runs:
            print("\nAvailable test runs:")
            print("-" * 50)
            for idx, run in enumerate(runs, 1):
                print(f"{idx}. Run ID: {run['run_id']}")
                print(f"   Start Time: {run['start_time']}")
                print(f"   End Time: {run['end_time']}")
                print(f"   Total Tests: {run['total_tests']}")
                print(f"   Success Rate: {run['success_rate']}")
                print()
        else:
            print(f"No test runs found in {base_dir}")

    elif args.command == "show-run":
        # Find the run directory
        run_dir = Path(base_dir) / args.run_id
        if not run_dir.exists():
            # Try to find by prefix
            candidates = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith(args.run_id)]
            if candidates:
                run_dir = candidates[0]
            else:
                print(f"Test run not found: {args.run_id}")
                return

        show_run_summary(str(run_dir))

    elif args.command == "list-iterations":
        # Find the run directory
        run_dir = Path(base_dir) / args.run_id
        if not run_dir.exists():
            # Try to find by prefix
            candidates = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith(args.run_id)]
            if candidates:
                run_dir = candidates[0]
            else:
                print(f"Test run not found: {args.run_id}")
                return

        iterations = list_test_iterations(str(run_dir), args.test_name)
        if iterations:
            print(f"\nIterations for test '{args.test_name}' in run '{args.run_id}':")
            print("-" * 50)
            for idx, iter_info in enumerate(iterations, 1):
                valid = "✅" if iter_info["is_valid"] else "❌"
                syntax = "❌" if iter_info["has_syntax_errors"] else "✅"

                # Handle processing success status safely
                processing_success = iter_info["processing_success"]
                if processing_success is not None:
                    exec_success = "✅" if processing_success else "❌"
                else:
                    exec_success = "⚠️"

                print(f"{idx}. Iteration {iter_info['iteration']}")
                print(f"   Timestamp: {iter_info['timestamp']}")
                print(f"   Response Time: {iter_info['response_time']}")
                print(f"   Valid KGML: {valid}")
                print(f"   Syntax Errors: {syntax}")
                print(f"   Processing Success: {exec_success}")
                print()
        else:
            print(f"No iterations found for test '{args.test_name}' in run '{args.run_id}'")

    elif args.command == "show-iteration":
        # Find the run directory
        run_dir = Path(base_dir) / args.run_id
        if not run_dir.exists():
            # Try to find by prefix
            candidates = [d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith(args.run_id)]
            if candidates:
                run_dir = candidates[0]
            else:
                print(f"Test run not found: {args.run_id}")
                return

        # Find the test directory
        test_dir = None
        for d in run_dir.iterdir():
            if d.is_dir() and (d.name == args.test_name or args.test_name.lower() in d.name.lower()):
                test_dir = d
                break

        if not test_dir:
            print(f"Test '{args.test_name}' not found in run: {args.run_id}")
            return

        # Find the iteration directory
        iter_name = f"iteration_{int(args.iteration):03d}"
        iter_dir = test_dir / iter_name
        if not iter_dir.exists():
            print(f"Iteration {args.iteration} not found for test '{args.test_name}' in run '{args.run_id}'")
            return

        show_iteration_details(str(iter_dir))


if __name__ == "__main__":
    main()
