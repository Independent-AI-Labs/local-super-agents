import ast
import logging
import re
from typing import Dict, Any, Optional

from integration.net.ollama.ollama_api import prompt_model


class KGMLInstructionParser:
    """
    Parser that converts natural language instructions within KGML commands
    to secure Python code for execution.

    This is the bridge between natural language in KGML commands and
    executable Python code that manipulates knowledge graph objects.
    """

    def __init__(self):
        """Initialize the parser with security settings."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.allowed_modules = ['math', 'datetime', 'json', 're']
        self.allowed_builtins = [
            'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool',
            'len', 'max', 'min', 'sum', 'sorted', 'enumerate', 'zip', 'range',
            'True', 'False', 'None'
        ]

    def parse_natural_language_instruction(self, instruction: str,
                                           entity_type: str, command_type: str,
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse a natural language instruction into executable Python code.

        Args:
            instruction: The natural language instruction
            entity_type: The type of entity being manipulated (NODE/LINK)
            command_type: The command being executed (C/U/D/E)
            context: Optional context information

        Returns:
            Dict with generated code and metadata
        """
        self.logger.info(f"Parsing instruction for {command_type}► {entity_type}: {instruction}")

        # Create appropriate system prompt based on command and entity type
        system_prompt = self._create_system_prompt(entity_type, command_type)

        # Prepare the prompt for the LLM
        prompt = self._create_llm_prompt(instruction, entity_type, command_type, context)

        # Get code from LLM
        try:
            response = prompt_model(
                message=prompt,
                model="qwen2.5-coder:14b",
                system_prompt=system_prompt
            )

            # Extract code from response
            code = self._extract_code_from_response(response)

            # Validate the code for security
            self._validate_code(code)

            return {
                "code": code,
                "original_instruction": instruction,
                "command_type": command_type,
                "entity_type": entity_type
            }
        except Exception as e:
            self.logger.error(f"Error parsing instruction: {str(e)}")
            raise ValueError(f"Failed to parse instruction: {str(e)}")

    def _create_system_prompt(self, entity_type: str, command_type: str) -> str:
        """
        Create an appropriate system prompt based on the command and entity type.
        """
        base_prompt = """
        You are a specialized code generator that converts natural language instructions 
        to secure Python code for knowledge graph operations.

        Your generated code MUST:
        1. Only manipulate the 'node' or 'edge' object provided to the function
        2. Use only basic Python operations (no imports except those explicitly allowed)
        3. Store results in the 'result' variable as a dictionary
        4. Be secure and not attempt to access any system resources
        5. Not use exec(), eval(), or any other dangerous functions
        6. Not import any modules except: math, datetime, json, re
        7. Not try to access filesystem, network, or environment variables
        8. Focus ONLY on the specific operation requested

        Focus on generating concise, effective code that accomplishes the instruction.
        """

        if entity_type == "NODE":
            if command_type == "C":
                # Create node command
                return base_prompt + """
                The 'node' variable is a KGNode object with these fields:
                - uid: Optional[str] - The node's unique identifier
                - type: Optional[str] - The node's type
                - meta_props: Dict[str, Any] - Dictionary of metadata properties
                - created_at: Optional[str] - Creation timestamp
                - updated_at: Optional[str] - Update timestamp

                Your code should set appropriate values for these fields based on the instruction.
                Remember to populate the 'result' variable with a dictionary containing at least
                {'status': 'created', 'node_id': node.uid}
                """
            elif command_type == "U":
                # Update node command
                return base_prompt + """
                The 'node' variable is an existing KGNode object you need to update based on the instruction.
                You can modify:
                - type: Optional[str] - The node's type
                - meta_props: Dict[str, Any] - Dictionary of metadata properties
                - updated_at: Optional[str] - Should be set to current time

                DO NOT modify the node's uid.
                Remember to populate the 'result' variable with a dictionary containing at least
                {'status': 'updated', 'node_id': node.uid}
                """
            elif command_type == "E":
                # Evaluate node command
                return base_prompt + """
                The 'node' variable is an existing KGNode object you need to evaluate.
                You can access:
                - node.uid: The node's unique identifier
                - node.type: The node's type
                - node.meta_props: Dictionary of metadata properties
                - node.content: The node's content (if it's a DataNode)

                Your code should compute a result based on the node's data and the instruction.
                Store your evaluation results in the 'result' variable.
                """
        elif entity_type == "LINK":
            if command_type == "C":
                # Create link command
                return base_prompt + """
                The 'edge' variable is a KGEdge object with these fields:
                - source_uid: str - The source node's uid
                - target_uid: str - The target node's uid
                - type: Optional[str] - The edge's type
                - meta_props: Dict[str, Any] - Dictionary of metadata properties
                - created_at: Optional[str] - Creation timestamp
                - updated_at: Optional[str] - Update timestamp

                Your code should set appropriate values for these fields based on the instruction.
                Remember to populate the 'result' variable with a dictionary containing at least
                {'status': 'created', 'edge': f"{edge.source_uid}->{edge.target_uid}"}
                """
            elif command_type == "U":
                # Update link command
                return base_prompt + """
                The 'edge' variable is an existing KGEdge object you need to update based on the instruction.
                You can modify:
                - type: Optional[str] - The edge's type
                - meta_props: Dict[str, Any] - Dictionary of metadata properties
                - updated_at: Optional[str] - Should be set to current time

                DO NOT modify the edge's source_uid or target_uid.
                Remember to populate the 'result' variable with a dictionary containing at least
                {'status': 'updated', 'edge': f"{edge.source_uid}->{edge.target_uid}"}
                """

        return base_prompt

    def _create_llm_prompt(self, instruction: str, entity_type: str,
                           command_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for the LLM based on the instruction and context.
        """
        command_names = {
            "C": "Create",
            "U": "Update",
            "D": "Delete",
            "E": "Evaluate"
        }

        entity_names = {
            "NODE": "node",
            "LINK": "edge"
        }

        cmd_name = command_names.get(command_type, command_type)
        entity_name = entity_names.get(entity_type, entity_type.lower())

        prompt = f"Generate Python code to {cmd_name.lower()} a knowledge graph {entity_name} based on this instruction:\n\n"
        prompt += f"\"{instruction}\"\n\n"

        # Add context if provided
        if context:
            prompt += "Additional context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"

        prompt += f"\nCreate code to {cmd_name.lower()} the {entity_name} according to the instruction. "
        prompt += f"Return only Python code within a code block that can be executed directly."

        return prompt

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from the LLM response.
        """
        # Try to find code blocks in markdown format
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # If no code blocks found, try to extract function definitions or code
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Check for lines that look like code
            if (line.strip().startswith('def ') or
                    line.strip().startswith('# ') or
                    line.strip().startswith('import ') or
                    line.strip().startswith('from ') or
                    '=' in line):
                in_code = True

            if in_code:
                code_lines.append(line)

            # End code block if we hit an empty line after code
            if in_code and not line.strip():
                in_code = False

        extracted_code = '\n'.join(code_lines).strip()
        if extracted_code:
            return extracted_code

        # As a fallback, use the entire response but warn
        self.logger.warning("Could not extract code blocks from LLM response, using raw response")
        return response.strip()

    def _validate_code(self, code: str) -> None:
        """
        Validate the generated code for security violations.
        Raises ValueError if the code is potentially unsafe.
        """
        # Parse the code to AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax in generated code: {e}")

        # Walk through the AST and check for security violations
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name not in self.allowed_modules:
                        raise ValueError(f"Disallowed import: {name.name}")

            # Check for import from
            elif isinstance(node, ast.ImportFrom):
                if node.module not in self.allowed_modules:
                    raise ValueError(f"Disallowed import from: {node.module}")

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', '__import__', 'compile']:
                        raise ValueError(f"Disallowed function call: {node.func.id}")

                # Check for attribute access like os.system
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ['os', 'subprocess', 'sys', 'builtins']:
                            raise ValueError(f"Disallowed module usage: {node.func.value.id}")

        # Additional security checks could be added here
