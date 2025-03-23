"""
Improved build and test view for the VibeCheck app - continued.
"""

from typing import Dict, List, Optional, Tuple

import gradio as gr

from ..controllers.test_controller import TestController


def create_build_test_tab(state: Dict) -> gr.Tab:
    """
    Create the build and test tab for the VibeCheck app with improved visual design.

    Args:
        state: Application state dictionary

    Returns:
        The build and test tab component
    """
    with gr.Tab("🧪 Build & Test") as build_test_tab:
        gr.Markdown(
            """
            # 🧪 Build & Test
            
            Discover, run, and visualize tests for your project. This tab helps you
            ensure your implementation meets the requirements and works as expected.
            """
        )
        
        with gr.Row():
            # Test discovery panel with improved styling
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Test Discovery")
                
                discover_tests_btn = gr.Button("🔍 Discover Tests", size="sm")
                
                discovered_tests = gr.CheckboxGroup(
                    label="Available Tests",
                    choices=[],
                    elem_id="discovered-tests"
                )
                
                with gr.Row():
                    run_selected_tests_btn = gr.Button("▶️ Run Selected", size="sm")
                    run_all_tests_btn = gr.Button("▶️ Run All Tests", size="sm")
            
            # Test results panel with improved styling
            with gr.Column(scale=2):
                with gr.Tabs() as test_tabs:
                    # Results tab with improved styling
                    with gr.Tab("📊 Results") as results_tab:
                        test_summary = gr.Dataframe(
                            headers=["🧪 Test", "📊 Status", "⏱️ Duration (s)"],
                            col_count=(3, "fixed"),
                            interactive=False,
                            elem_id="test-summary"
                        )
                        
                        with gr.Row():
                            # Use different colors for stats
                            with gr.Column():
                                passed_count = gr.Number(
                                    label="✅ Passed",
                                    value=0,
                                    interactive=False,
                                    elem_id="passed-count"
                                )
                            
                            with gr.Column():
                                failed_count = gr.Number(
                                    label="❌ Failed",
                                    value=0,
                                    interactive=False,
                                    elem_id="failed-count"
                                )
                            
                            with gr.Column():
                                pass_rate = gr.Number(
                                    label="📈 Pass Rate (%)",
                                    value=0,
                                    interactive=False,
                                    elem_id="pass-rate"
                                )
                        
                        execution_time = gr.Textbox(
                            label="⏱️ Execution Time",
                            value="",
                            interactive=False
                        )
                    
                    # Output tab with improved styling
                    with gr.Tab("📝 Output") as output_tab:
                        test_output = gr.TextArea(
                            label="🖥️ Test Output",
                            value="",
                            interactive=False,
                            lines=25,
                            elem_id="test-output"
                        )
                    
                    # Visualization tab with improved styling
                    with gr.Tab("📈 Visualization") as viz_tab:
                        visualization = gr.HTML(
                            label="📊 Test Visualization",
                            elem_id="test-visualization"
                        )
        
        # Function to discover tests
        def discover_tests() -> List[str]:
            if not state.get("current_project"):
                return []
            
            project_path = state["current_project"].metadata.path
            tests = TestController.discover_tests(project_path)
            
            # Add emoji to test names
            return [f"🧪 {test}" for test in tests]
        
        # Function to run tests
        def run_tests(selected_tests: List[str]) -> Tuple[List[List[str]], float, float, float, str, str, str]:
            if not state.get("current_project"):
                return [], 0, 0, 0, "", "", ""
            
            if not selected_tests:
                return [], 0, 0, 0, "", "", ""
            
            # Remove emoji from test names
            cleaned_tests = [test.split(' ', 1)[1] if ' ' in test else test for test in selected_tests]
            
            project_path = state["current_project"].metadata.path
            test_result = TestController.run_tests(project_path, cleaned_tests)
            
            # Prepare summary data
            summary_data = []
            for test in test_result.tests:
                # Add status emoji
                status_emoji = "✅" if test.status == "passed" else "❌" if test.status == "failed" else "⚠️" if test.status == "error" else "⏭️"
                status_display = f"{status_emoji} {test.status}"
                
                summary_data.append([
                    test.name,
                    status_display,
                    f"{test.duration:.2f}"
                ])
            
            # Calculate pass rate
            pass_rate = 0
            if len(test_result.tests) > 0:
                pass_rate = (test_result.passed_count / len(test_result.tests)) * 100
            
            # Format execution time
            execution_time_str = test_result.execution_date.strftime("%Y-%m-%d %H:%M:%S")
            
            # Combine all test outputs
            combined_output = "\n\n".join(f"### 🧪 {test.name}\n\n{test.output}" for test in test_result.tests)
            
            # Generate visualization HTML
            viz_data = TestController.visualize_test_results(test_result)
            visualization_html = generate_visualization_html(viz_data)
            
            return (
                summary_data,
                test_result.passed_count,
                test_result.failed_count,
                pass_rate,
                execution_time_str,
                combined_output,
                visualization_html
            )
        
        # Function to generate HTML for test visualization with improved styling
        def generate_visualization_html(viz_data: Dict) -> str:
            """Generate HTML for test visualization with modern styling."""
            status_colors = {
                "passed": "#4CAF50",  # Green
                "failed": "#F44336",  # Red
                "skipped": "#9E9E9E",  # Gray
                "error": "#FF9800"    # Orange
            }
            
            status_emojis = {
                "passed": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "error": "⚠️"
            }
            
            html = f"""
            <div style="padding: 20px; font-family: Arial, sans-serif; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px;">📊 Test Results: {viz_data['name']}</h2>
                <p style="color: #666;">🕒 Execution Date: {viz_data['execution_date']}</p>
                
                <div style="margin: 20px 0; display: flex; gap: 20px; flex-wrap: wrap;">
                    <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #2E7D32;">✅ Passed</h3>
                        <p style="font-size: 24px; margin: 0; color: #4CAF50; font-weight: bold;">{viz_data['status_counts']['passed']}</p>
                    </div>
                    <div style="background-color: #FFEBEE; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #C62828;">❌ Failed</h3>
                        <p style="font-size: 24px; margin: 0; color: #F44336; font-weight: bold;">{viz_data['status_counts']['failed']}</p>
                    </div>
                    <div style="background-color: #EEEEEE; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #616161;">⏭️ Skipped</h3>
                        <p style="font-size: 24px; margin: 0; color: #9E9E9E; font-weight: bold;">{viz_data['status_counts']['skipped']}</p>
                    </div>
                    <div style="background-color: #FFF3E0; padding: 15px; border-radius: 8px; min-width: 150px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <h3 style="margin: 0 0 10px 0; color: #E65100;">⚠️ Error</h3>
                        <p style="font-size: 24px; margin: 0; color: #FF9800; font-weight: bold;">{viz_data['status_counts']['error']}</p>
                    </div>
                </div>
                
                <div style="margin: 30px 0; background-color: #f5f5f5; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="margin-top: 0;">📈 Pass Rate</h3>
                    <div style="width: 100%; background-color: #EEEEEE; height: 25px; border-radius: 5px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="width: {viz_data['pass_percentage']}%; background-color: #4CAF50; height: 100%;"></div>
                    </div>
                    <p style="text-align: center; margin-top: 5px; font-weight: bold;">{viz_data['pass_percentage']:.1f}%</p>
                </div>
                
                <h3 style="color: #333; margin-top: 30px;">📋 Test Results</h3>
                <div style="overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <table style="width: 100%; border-collapse: collapse; background-white;">
                        <thead>
                            <tr style="background-color: #f5f5f5;">
                                <th style="text-align: left; padding: 12px; border-bottom: 2px solid #ddd;">🧪 Test</th>
                                <th style="text-align: left; padding: 12px; border-bottom: 2px solid #ddd;">📊 Status</th>
                                <th style="text-align: left; padding: 12px; border-bottom: 2px solid #ddd;">⏱️ Duration (s)</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for test in viz_data['test_results']:
                color = status_colors.get(test['status'], "#000000")
                emoji = status_emojis.get(test['status'], "")
                
                html += f"""
                    <tr style="border-bottom: 1px solid #ddd; transition: background-color 0.2s;">
                        <td style="padding: 12px;">{test['name']}</td>
                        <td style="padding: 12px;">
                            <span style="color: {color}; font-weight: bold;">{emoji} {test['status']}</span>
                        </td>
                        <td style="padding: 12px;">{test['duration']:.2f}</td>
                    </tr>
                """
            
            html += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
            
            return html
        
        # Connect event handlers
        discover_tests_btn.click(
            discover_tests,
            inputs=None,
            outputs=[discovered_tests]
        )
        
        run_selected_tests_btn.click(
            run_tests,
            inputs=[discovered_tests],
            outputs=[
                test_summary,
                passed_count,
                failed_count,
                pass_rate,
                execution_time,
                test_output,
                visualization
            ]
        )
        
        run_all_tests_btn.click(
            lambda: run_tests(discover_tests()),
            inputs=None,
            outputs=[
                test_summary,
                passed_count,
                failed_count,
                pass_rate,
                execution_time,
                test_output,
                visualization
            ]
        )
        
        # When the Build & Test tab is clicked, discover tests
        build_test_tab.select(
            discover_tests,
            inputs=None,
            outputs=[discovered_tests]
        )
    
    return build_test_tab
