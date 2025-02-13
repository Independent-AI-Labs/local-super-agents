import csv
import math
import os
import zipfile

from io import StringIO
from typing import Dict
from typing import List, Optional

import cpuinfo
import psutil

from tabulate import tabulate

from retrieval.hype.data.models import UnifiedSearchResult


def deploy_test_resources():
    print("Preparing benchmark data, this might take a few minutes...")

    zip_prefix = os.path.join("res", "test_resources.zip.")

    # N number of parts
    import glob

    parts = glob.glob(zip_prefix + '*')
    n = len(parts)

    # Concatenate
    with open(os.path.join("res", "test_resources.zip"), "wb") as outfile:
        for i in range(1, n + 1):
            filename = zip_prefix + str(i).zfill(3)
            with open(filename, "rb") as infile:
                outfile.write(infile.read())

    with zipfile.ZipFile(os.path.join("res", "test_resources.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join('res', 'test_resources'))

    # Delete temporary concat file.
    os.remove(os.path.join("res", "test_resources.zip"))

    print(f"Successfully deployed latest test data. Commencing tests...")


def get_system_specs() -> Dict[str, str]:
    platform_cpu = {}

    platform_cpu["name"] = cpuinfo.get_cpu_info()["brand_raw"]
    platform_cpu["core_cnt"] = psutil.cpu_count(logical=True)
    platform_cpu["phy_core_cnt"] = psutil.cpu_count(logical=False)
    cpufreq = psutil.cpu_freq()
    platform_cpu["proc_freq_mhz"] = round(cpufreq.max, 2)
    platform_cpu["proc_freq_curr_mhz"] = round(cpufreq.current, 2)
    platform_cpu["proc_load"] = psutil.cpu_percent()

    # Needs sudo, so a no-go for now...
    # platform_cpu[mem_module_cnt = subprocess.check_output(["sudo dmidecode --type memory"])....
    vmem = psutil.virtual_memory()

    platform_cpu["mem_cap_gb"] = math.ceil(float(vmem.total) / (1024 * 1024 * 1024))
    platform_cpu["mem_alloc_gb"] = math.ceil(float(vmem.used) / (1024 * 1024 * 1024))
    platform_cpu["mem_alloc_percent"] = round(platform_cpu["mem_alloc_gb"] / platform_cpu["mem_cap_gb"] * 100, 1)

    return platform_cpu


def draw_ascii_bar(value: float, max_value: float, bar_length: int = 50) -> str:
    """
    Draws an ASCII bar representing the value proportionally to max_value.
    """
    proportion = value / max_value
    filled_length = int(round(bar_length * proportion))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    return f"|{bar}| {value:.1f}%"


def check_system_specs() -> str:
    output = ""

    # Assuming get_system_specs() returns a dictionary with system specs
    specs = get_system_specs()

    output += f"CPU: {specs['name']}\n"
    output += f"Total / Logical CPU Cores: {specs['core_cnt']}\n"
    output += f"Physical CPU Cores: {specs['phy_core_cnt']}\n"

    output += "\n"

    output += f"Current CPU Frequency: {specs['proc_freq_curr_mhz']} MHz\n"
    output += f"CPU Load: {specs['proc_load']}%\n"
    output += draw_ascii_bar(float(specs['proc_load']), 100) + "\n"

    output += "\n"

    output += f"Total Memory: {specs['mem_cap_gb']} GB\n"
    output += f"Allocated Memory: {specs['mem_alloc_gb']} GB ({specs['mem_alloc_percent']}%)\n"
    output += draw_ascii_bar(float(specs['mem_alloc_percent']), 100) + "\n"

    output += "\n"

    # Print and also return the output string
    print(output)
    return output


def print_search_results(
        search_results: List[UnifiedSearchResult],
        top_n: int = 10,
        headers: Optional[List[str]] = None
) -> None:
    """
    Print the top N search results using the UnifiedSearchResult model.
    Handles both file matches and structured matches.

    Args:
        search_results (List[UnifiedSearchResult]): List of UnifiedSearchResult objects.
        top_n (int): Number of top results to print (default: 10).
        headers (Optional[List[str]]): Optional list of column names for structured data (default: None).
    """
    for result in search_results[:top_n]:
        print("---")
        if result.file_matches:
            print(f"URI: {result.common.uri} ({len(result.common.matched_search_terms)} Hits)\n")
            print(result.file_matches.title)
            print("---")
            print("Matches:\n")
            for match in result.file_matches.matches_with_context:
                print("···")
                print(match[:256])
            print("······")
        elif result.structured_matches:
            print(f"URI: {result.common.uri}")
            print(f"Item Index: {result.structured_matches.item_index}")
            print("---")

            # print("Item Content RAW:")
            # print(result.structured_matches.item_content)

            print()
            print("Item Content:")
            csv_reader = csv.reader(StringIO(result.structured_matches.item_content))
            row_data = next(csv_reader)

            if headers:
                table_headers = headers[:len(row_data)]
                if len(row_data) > len(table_headers):
                    table_headers.extend([f"Column {i + 1}" for i in range(len(table_headers), len(row_data))])
            else:
                table_headers = [f"Column {i + 1}" for i in range(len(row_data))]

            table = [table_headers, row_data]

            print(tabulate(table, headers="firstrow", tablefmt="grid"))

        print("\nSearch Term Match Counts:")
        print(result.common.search_term_match_counts)
        print(f"\nTotal Matches: {len(result.common.matched_search_terms)}")
        print(f"\nScore: {result.common.score}")
        print("-------\n")
