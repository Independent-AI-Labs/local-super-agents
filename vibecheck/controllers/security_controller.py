"""
Security analysis controller for VibeCheck.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from vibecheck.models.security import SecurityVulnerability, SecurityAnalysis
from vibecheck.integrations.betterscan import scan_project
from vibecheck.integrations.llm import analyze_security_vulnerabilities
from vibecheck.integrations.web import research_security_patterns
from vibecheck.utils.cache_utils import AnalysisCache
from vibecheck.utils.file_utils import ensure_directory, read_file, write_file
from vibecheck import config


class SecurityController:
    """
    Controller for security analysis functions including integration with
    Betterscan and LLM-based security insights.
    """

    @staticmethod
    def run_security_analysis(project_path: str, target_path: str = None) -> List[SecurityAnalysis]:
        """
        Run a security analysis on the project or a specific path within it.

        Args:
            project_path: Path to the project
            target_path: Optional specific path to analyze (relative to project_path)

        Returns:
            List of SecurityAnalysis results
        """
        # Check cache first
        cache_key = "security_analysis_" + (target_path or "full_project")
        cached_results = AnalysisCache.get_cached_analysis(project_path, cache_key, "security")
        
        if cached_results:
            try:
                return [SecurityAnalysis.parse_obj(item) for item in cached_results]
            except Exception as e:
                print(f"Error parsing cached security analysis: {e}")
        
        path_to_analyze = project_path
        if target_path:
            path_to_analyze = os.path.join(project_path, target_path)
        
        # Run Betterscan on the path
        vulnerabilities_by_path: Dict[str, List[SecurityVulnerability]] = scan_project(path_to_analyze)
        
        # Create security analysis results
        results: List[SecurityAnalysis] = []
        now = datetime.now()
        
        for path, vulnerabilities in vulnerabilities_by_path.items():
            if not vulnerabilities:
                continue
                
            # Get LLM insights for the vulnerabilities
            insights = SecurityController._get_llm_insights(vulnerabilities)
            
            # Enhance with web research for major vulnerabilities
            if any(v.severity in ["critical", "high"] for v in vulnerabilities):
                web_insights = SecurityController._get_web_research_insights(path, vulnerabilities)
                if web_insights:
                    insights += "\n\n## Additional Research Insights\n\n" + web_insights
            
            # Create security analysis
            analysis = SecurityAnalysis(
                path=path,
                vulnerabilities=vulnerabilities,
                analysis_date=now,
                llm_insights=insights
            )
            
            results.append(analysis)
            
            # Save the analysis
            SecurityController._save_security_analysis(project_path, analysis)
        
        # Cache the results
        if results:
            AnalysisCache.cache_analysis(
                project_path,
                cache_key,
                "security",
                [json.loads(item.json()) for item in results],
                ttl_seconds=3600  # 1 hour
            )
        
        return results

    @staticmethod
    def get_project_security_summary(project_path: str) -> Dict:
        """
        Get a summary of security vulnerabilities for the project.

        Args:
            project_path: Path to the project

        Returns:
            Dictionary with security summary
        """
        # Get all security analyses for the project
        security_dir = os.path.join(project_path, config.IMPLEMENTATION_SECURITY_DIR)
        
        if not os.path.exists(security_dir):
            return {
                "total_vulnerabilities": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "info": 0
                },
                "last_analysis": None,
                "high_priority_files": []
            }
        
        # Load all security analyses
        analyses = []
        
        for filename in os.listdir(security_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(security_dir, filename), 'r') as f:
                        data = json.load(f)
                        analyses.append(SecurityAnalysis.parse_obj(data))
                except Exception as e:
                    print(f"Error loading security analysis {filename}: {e}")
        
        if not analyses:
            return {
                "total_vulnerabilities": 0,
                "severity_counts": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "info": 0
                },
                "last_analysis": None,
                "high_priority_files": []
            }
        
        # Calculate summary
        total_vulnerabilities = 0
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        # Track files by issue severity
        files_by_severity = {
            "critical": set(),
            "high": set(),
            "medium": set(),
            "low": set(),
            "info": set()
        }
        
        last_analysis = max(a.analysis_date for a in analyses)
        
        for analysis in analyses:
            for vuln in analysis.vulnerabilities:
                total_vulnerabilities += 1
                severity = vuln.severity.lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
                    files_by_severity[severity].add(analysis.path)
        
        # Generate list of high priority files (with critical or high vulnerabilities)
        high_priority_files = list(files_by_severity["critical"] | files_by_severity["high"])
        
        return {
            "total_vulnerabilities": total_vulnerabilities,
            "severity_counts": severity_counts,
            "last_analysis": last_analysis.isoformat() if last_analysis else None,
            "high_priority_files": high_priority_files
        }

    @staticmethod
    def get_vulnerability_trends(project_path: str, days: int = 30) -> Dict:
        """
        Get trends in security vulnerabilities over time.

        Args:
            project_path: Path to the project
            days: Number of days to analyze

        Returns:
            Dictionary with vulnerability trends
        """
        # This would typically use historical security analysis data
        # For now, we'll return a placeholder
        return {
            "trend": "stable",
            "new_vulnerabilities": 0,
            "fixed_vulnerabilities": 0,
            "days_analyzed": days
        }

    @staticmethod
    def _get_llm_insights(vulnerabilities: List[SecurityVulnerability]) -> str:
        """
        Get LLM-generated insights for the given vulnerabilities.

        Args:
            vulnerabilities: List of vulnerabilities to analyze

        Returns:
            LLM-generated insights as text
        """
        if not vulnerabilities:
            return "No vulnerabilities detected. The code appears to be secure based on the current analysis."
        
        # Format vulnerabilities for LLM analysis
        vulnerabilities_text = "## Security Vulnerabilities\n\n"
        
        for i, vuln in enumerate(vulnerabilities):
            vulnerabilities_text += f"### Vulnerability {i+1}: {vuln.severity.upper()}\n"
            vulnerabilities_text += f"**Description**: {vuln.description}\n"
            vulnerabilities_text += f"**Location**: {vuln.location}\n"
            vulnerabilities_text += f"**Recommendation**: {vuln.recommendation}\n\n"
        
        # Get LLM analysis
        try:
            insights = analyze_security_vulnerabilities(vulnerabilities_text)
            return insights
        except Exception as e:
            print(f"Error getting LLM insights: {e}")
            
            # Generate a fallback analysis based on the vulnerabilities
            severity_counts = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "info": 0
            }
            
            for vuln in vulnerabilities:
                if vuln.severity in severity_counts:
                    severity_counts[vuln.severity] += 1
            
            insights = [
                "## Security Analysis Summary",
                "",
                "### Vulnerability Overview",
                f"- Critical: {severity_counts['critical']}",
                f"- High: {severity_counts['high']}",
                f"- Medium: {severity_counts['medium']}",
                f"- Low: {severity_counts['low']}",
                f"- Info: {severity_counts['info']}",
                "",
                "### Key Findings"
            ]
            
            if severity_counts["critical"] > 0 or severity_counts["high"] > 0:
                insights.append("- **High Priority Issues**: There are critical/high severity issues that require immediate attention.")
            else:
                insights.append("- No critical or high severity issues were detected in this analysis.")
            
            insights.append("")
            insights.append("### Recommendations")
            insights.append("1. Review and address the identified vulnerabilities according to their severity.")
            insights.append("2. Consider implementing automated security scanning in your CI/CD pipeline.")
            insights.append("3. Conduct regular security training for the development team.")
            
            return "\n".join(insights)

    @staticmethod
    def _get_web_research_insights(file_path: str, vulnerabilities: List[SecurityVulnerability]) -> str:
        """
        Get additional insights from web research for the given vulnerabilities.

        Args:
            file_path: Path to the file with vulnerabilities
            vulnerabilities: List of vulnerabilities to research

        Returns:
            Web research insights as text
        """
        # Extract component name from file path
        component_name = os.path.basename(file_path).split('.')[0]
        
        # Get security patterns for the component
        try:
            filtered_data, discovered_patterns = research_security_patterns(component_name)
            
            if not filtered_data:
                return ""
            
            # Format insights
            insights = "### Security Research Findings\n\n"
            
            # Focus on relevant findings
            relevant_findings = []
            for i, finding in enumerate(filtered_data[:5]):  # Limit to top 5 findings
                title = finding.get("title", f"Finding {i+1}")
                content = finding.get("content", "").strip()
                
                if content:
                    # Only include if it seems relevant
                    for vuln in vulnerabilities:
                        if (vuln.severity.lower() in content.lower() or 
                            vuln.description.lower() in content.lower() or
                            any(p.lower() in content.lower() for p in discovered_patterns)):
                            relevant_findings.append((title, content))
                            break
            
            if not relevant_findings:
                return ""
            
            for title, content in relevant_findings:
                insights += f"#### {title}\n\n"
                # Limit content length to keep it manageable
                if len(content) > 500:
                    content = content[:500] + "..."
                insights += f"{content}\n\n"
            
            return insights
        
        except Exception as e:
            print(f"Error getting web research insights: {e}")
            return ""

    @staticmethod
    def load_security_analysis(project_path: str, target_path: str) -> Optional[SecurityAnalysis]:
        """
        Load security analysis results for a specific path.

        Args:
            project_path: Path to the project
            target_path: Target path for which to load the analysis (relative to project_path)

        Returns:
            SecurityAnalysis or None if not found
        """
        # Normalize target path
        normalized_path = target_path.replace('/', '_').replace('\\', '_')
        
        analysis_path = os.path.join(
            project_path,
            config.IMPLEMENTATION_SECURITY_DIR,
            f"{normalized_path}.json"
        )
        
        if not os.path.exists(analysis_path):
            return None
        
        try:
            with open(analysis_path, 'r') as f:
                data = json.load(f)
                return SecurityAnalysis.parse_obj(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading security analysis: {e}")
            return None

    @staticmethod
    def _save_security_analysis(project_path: str, analysis: SecurityAnalysis) -> None:
        """
        Save security analysis results to the project.

        Args:
            project_path: Path to the project
            analysis: SecurityAnalysis to save
        """
        # Ensure the directory exists
        security_dir = os.path.join(project_path, config.IMPLEMENTATION_SECURITY_DIR)
        ensure_directory(security_dir)
        
        # Normalize the path
        normalized_path = analysis.path.replace('/', '_').replace('\\', '_')
        
        # Save the analysis
        analysis_path = os.path.join(security_dir, f"{normalized_path}.json")
        with open(analysis_path, 'w') as f:
            f.write(analysis.json(indent=2))
