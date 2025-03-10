"""
KGML Test Helpers

This module contains helper functions and classes for KGML integration tests.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

# Import the new parser class and tokenizer
from knowledge.reasoning.dsl.kgml_parser import Parser, tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KGMLIntegrationTests")


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def is_plain_text(prompt: str) -> bool:
    """
    Validate that the prompt is a plain-text string.
    We check that it does not start with '{' (indicating JSON)
    and that it contains expected KGML markers.
    """
    return (not prompt.strip().startswith("{") and
            re.search(r'(KG►|KGNODE►|KGLINK►|C►|U►|D►|E►|N►)', prompt) is not None)


def validate_kgml_with_error(kgml_text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the provided text is valid KGML by attempting to parse it.
    Returns a tuple of (is_valid, error_message) where error_message is None if valid.
    """
    try:
        tokens = tokenize(kgml_text)
        parser = Parser(tokens)
        _ = parser.parse_program()
        return True, None
    except Exception as e:
        return False, str(e)


def validate_kgml(kgml_text: str) -> bool:
    """
    Validate that the provided text is valid KGML by attempting to parse it.
    Returns True if parsing succeeds, False otherwise.
    """
    is_valid, _ = validate_kgml_with_error(kgml_text)
    return is_valid


def format_reasoning_summary(evaluation_results: List[Dict[str, Any]], difficulty_levels: List[str]) -> str:
    """
    Format a summary of reasoning evaluation results.
    """
    summary = ["## Reasoning Evaluation Summary"]

    # Overall statistics
    total_problems = len(evaluation_results)
    successful_problems = sum(1 for r in evaluation_results if r["goal_reached"])

    summary.append(f"Total problems: {total_problems}")
    summary.append(f"Successfully solved: {successful_problems} ({successful_problems / total_problems * 100:.1f}%)")

    # Results by difficulty level
    for difficulty in difficulty_levels:
        problems_at_level = [r for r in evaluation_results if r["difficulty"] == difficulty]
        if problems_at_level:
            success_at_level = sum(1 for r in problems_at_level if r["goal_reached"])
            success_rate = success_at_level / len(problems_at_level) * 100
            summary.append(f"{difficulty.capitalize()} problems: {success_at_level}/{len(problems_at_level)} ({success_rate:.1f}%)")

    # Individual problem results
    summary.append("\n### Detailed Results")
    for result in evaluation_results:
        status = "✅ SOLVED" if result["goal_reached"] else "❌ FAILED"
        iterations = result["iterations"]
        summary.append(f"{result['problem_id']} ({result['difficulty']}): {status} in {iterations} iterations")

    return "\n".join(summary)
