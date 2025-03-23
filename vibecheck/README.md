# VibeCheck

## 🚀 Overview

VibeCheck is a Gradio-based tool designed to address the challenges of "vibe coding" - the practice of using Large Language Models (LLMs) to generate extensive amounts of code with-or-without proper software engineering principles. By enforcing architectural design, implementation tracking, security analysis, environment management, and testing, VibeCheck helps developers maintain high-quality standards while leveraging the productivity benefits of AI.

## 🎯 Key Features

- **📐 Architecture Designer**: Define your software architecture before implementation
  - Rich text editor for architecture documentation
  - Automated diagram generation (module, dataflow, security)
  - Architecture analysis using LLMs

- **📊 Implementation Tracker**: Monitor your development progress
  - Git integration for change tracking
  - Implementation percentage calculation
  - Source code viewer and editor

- **🔒 Security Analyzer**: Identify vulnerabilities in your code
  - Integration with Betterscan for security scanning
  - LLM-enhanced security insights
  - Web research for security patterns

- **🧰 Environment Manager**: Manage your development environments
  - Dependency management
  - Virtual environment configuration
  - Environment variable management

- **🧪 Test Runner**: Ensure code quality through testing
  - Test discovery and execution
  - Test results visualization
  - LLM-powered test generation

## 📋 Installation

### Prerequisites

- Python 3.11+
- Git (for implementation tracking)
- Betterscan CLI (for security analysis)

### Installation Steps
Run the appropriate *setup* script (Windows and Linux supported).

Navigate to `http://localhost:7860` in your web browser to access the application.


## 🏛️ Architecture

VibeCheck follows the Model-View-Controller (MVC) architectural pattern:

- **Models**: Pydantic models for data validation and representation
- **Views**: Gradio UI components for user interaction
- **Controllers**: Business logic for each functional component
- **Integrations**: External services and tools integration
- **Utils**: Common utility functions

```
vibecheck/
├── __init__.py
├── app.py               # Main application entry point
├── config.py            # Global constants and configuration
├── models/              # Pydantic models
├── controllers/         # Business logic
├── views/               # Gradio UI components
├── integrations/        # External integrations
└── utils/               # Utility functions
```