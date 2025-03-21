"""
KGML Code Generator Interface - Defines interfaces for code generation.

This module provides a consistent interface for generating Python code
from natural language instructions in the KGML system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class CodeGenerationError(Exception):
    """Exception raised for errors during code generation."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class CodeGenerator(ABC):
    """
    Abstract base class for code generators.
    
    Defines a common interface for generating Python code from
    natural language instructions.
    """
    
    @abstractmethod
    def generate_code(self, 
                      instruction: str, 
                      entity_type: str, 
                      command_type: str, 
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Python code from a natural language instruction.
        
        Args:
            instruction: The natural language instruction
            entity_type: The type of entity (NODE, LINK)
            command_type: The type of command (C, U, D, E)
            context: Optional additional context information
            
        Returns:
            Generated Python code as a string
            
        Raises:
            CodeGenerationError: If code generation fails
        """
        pass
    
    @abstractmethod
    def validate_code(self, code: str) -> None:
        """
        Validate generated code for security and correctness.
        
        Args:
            code: The code to validate
            
        Raises:
            CodeGenerationError: If the code is invalid or potentially unsafe
        """
        pass


class LLMCodeGenerator(CodeGenerator):
    """
    Code generator that uses a large language model (LLM) to generate code.
    
    This class encapsulates the logic for prompting an LLM to generate
    Python code from natural language instructions.
    """
    
    def __init__(self, 
                 model_id: str = "qwen2.5-coder:14b", 
                 allowed_modules: Optional[List[str]] = None, 
                 allowed_builtins: Optional[List[str]] = None):
        """
        Initialize the LLM code generator.
        
        Args:
            model_id: The LLM model ID to use
            allowed_modules: List of allowed module imports
            allowed_builtins: List of allowed built-in functions
        """
        self.model_id = model_id
        self.allowed_modules = allowed_modules or ['math', 'datetime', 'json', 're']
        self.allowed_builtins = allowed_builtins or [
            'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool',
            'len', 'max', 'min', 'sum', 'sorted', 'enumerate', 'zip', 'range',
            'True', 'False', 'None'
        ]
        
    def generate_code(self, 
                     instruction: str, 
                     entity_type: str, 
                     command_type: str, 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Python code using an LLM.
        
        Args:
            instruction: The natural language instruction
            entity_type: The type of entity (NODE, LINK)
            command_type: The type of command (C, U, D, E)
            context: Optional additional context information
            
        Returns:
            Generated Python code as a string
            
        Raises:
            CodeGenerationError: If code generation fails
        """
        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(entity_type, command_type, context)
            
            # Create user prompt
            user_prompt = self._create_user_prompt(instruction, entity_type, command_type, context)
            
            # Get response from LLM
            code = self._prompt_llm(system_prompt, user_prompt)
            
            # Extract and validate code
            extracted_code = self._extract_code(code)
            self.validate_code(extracted_code)
            
            return extracted_code
        except Exception as e:
            raise CodeGenerationError(f"Failed to generate code: {str(e)}")
    
    def validate_code(self, code: str) -> None:
        """
        Validate generated code for security and correctness.
        
        Args:
            code: The code to validate
            
        Raises:
            CodeGenerationError: If the code is invalid or potentially unsafe
        """
        import ast
        
        # Parse the code to AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeGenerationError(f"Invalid Python syntax: {e}")
            
        # Walk through the AST and check for security violations
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name not in self.allowed_modules:
                        raise CodeGenerationError(f"Disallowed import: {name.name}")
                        
            # Check for import from
            elif isinstance(node, ast.ImportFrom):
                if node.module not in self.allowed_modules:
                    raise CodeGenerationError(f"Disallowed import from: {node.module}")
                    
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', '__import__', 'compile']:
                        raise CodeGenerationError(f"Disallowed function call: {node.func.id}")
                        
                # Check for attribute access like os.system
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id in ['os', 'subprocess', 'sys', 'builtins']:
                            raise CodeGenerationError(f"Disallowed module usage: {node.func.value.id}")
    
    def _create_system_prompt(self, 
                              entity_type: str, 
                              command_type: str, 
                              context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a system prompt for the LLM based on entity and command type.
        
        Args:
            entity_type: The type of entity (NODE, LINK)
            command_type: The type of command (C, U, D, E)
            context: Optional additional context information
            
        Returns:
            System prompt string
        """
        base_prompt = """
        You are a specialized code generator that converts natural language instructions 
        to secure Python code for knowledge graph operations.

        Your generated code MUST:
        1. Only manipulate the provided object ('node' or 'edge')
        2. Use only basic Python operations (no imports except those explicitly allowed)
        3. Store results in the 'result' variable as a dictionary
        4. Be secure and not attempt to access any system resources
        5. Not use exec(), eval(), or any other dangerous functions
        6. Not import any modules except: math, datetime, json, re
        7. Not try to access filesystem, network, or environment variables
        8. Focus ONLY on the specific operation requested

        RETURN ONLY THE PYTHON CODE - NO EXPLANATIONS OR MARKDOWN.
        """
        
        # Add entity-specific and command-specific instructions
        if entity_type == "NODE":
            if command_type == "C":  # Create
                return base_prompt + """
                The 'node' variable is a KGNode object you need to populate based on the instruction.
                You can set:
                - node.type: str - The node's type
                - node.meta_props: Dict[str, Any] - Dictionary of metadata properties

                Remember to populate the 'result' variable with a dictionary containing at least:
                {'status': 'created', 'node_id': node.uid, 'node_type': node.type}
                """
            elif command_type == "U":  # Update
                return base_prompt + """
                The 'node' variable is an existing KGNode object you need to update based on the instruction.
                You can modify:
                - node.meta_props: Dict[str, Any] - Dictionary of metadata properties
                - Any other specific properties the node type might have

                DO NOT modify the node's uid or type.

                Remember to populate the 'result' variable with a dictionary containing at least:
                {'status': 'updated', 'node_id': node.uid}
                """
            elif command_type == "E":  # Evaluate
                return base_prompt + """
                The 'node' variable is an existing KGNode object you need to evaluate.
                You can access:
                - node.uid: The node's unique identifier
                - node.type: The node's type
                - node.meta_props: Dictionary of metadata properties
                - Any other specific properties the node type might have

                Your code should compute a result based on the node's data and the instruction.
                Store your evaluation results in the 'result' variable.
                """
            elif command_type == "D":  # Delete
                return base_prompt + """
                The 'node' variable is an existing KGNode object that will be deleted.
                You should perform any necessary cleanup or validation before deletion.

                Remember to populate the 'result' variable with a dictionary containing at least:
                {'status': 'deleting', 'node_id': node.uid}
                """
        elif entity_type == "LINK":
            if command_type == "C":  # Create
                return base_prompt + """
                The 'edge' variable is a KGEdge object with these fields:
                - source_uid: str - The source node's uid (ALREADY SET)
                - target_uid: str - The target node's uid (ALREADY SET)
                - relation: str - The edge's relation type (YOU NEED TO SET THIS)
                - meta_props: Dict[str, Any] - Dictionary of metadata properties

                IMPORTANT: The source_uid and target_uid are ALREADY SET. DO NOT try to extract or set them.
                
                Focus on determining the appropriate RELATION TYPE from the instruction.
                Valid relation types include: REPLY_TO, PART_OF, DERIVES_FROM, AUGMENTS, REFERENCES, etc.

                Remember to populate the 'result' variable with a dictionary containing at least:
                {'status': 'created', 'edge': f"{edge.source_uid}->{edge.target_uid}", 'relation': edge.relation}
                """
            elif command_type == "U":  # Update
                return base_prompt + """
                The 'edge' variable is an existing KGEdge object you need to update based on the instruction.
                You can modify:
                - edge.relation: str - The edge's relation type
                - edge.meta_props: Dict[str, Any] - Dictionary of metadata properties

                DO NOT modify the edge's source_uid or target_uid.

                Remember to populate the 'result' variable with a dictionary containing at least:
                {'status': 'updated', 'edge': f"{edge.source_uid}->{edge.target_uid}"}
                """
        
        # Special case for extracting link endpoints
        if context and context.get("parsing_mode") == "extract_link_endpoints":
            return base_prompt + """
            SPECIAL TASK: Extract the source node ID and target node ID from a natural language instruction.
            The instruction describes a link between two nodes. Your goal is to identify which node is the source
            and which is the target.

            Your code should:
            1. Parse the instruction to identify source and target
            2. Set result = {"source": "source_node_id", "target": "target_node_id"}

            Return the node IDs without any additional text.
            """
            
        return base_prompt
    
    def _create_user_prompt(self, 
                           instruction: str, 
                           entity_type: str, 
                           command_type: str, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a user prompt for the LLM.
        
        Args:
            instruction: The natural language instruction
            entity_type: The type of entity (NODE, LINK)
            command_type: The type of command (C, U, D, E)
            context: Optional additional context information
            
        Returns:
            User prompt string
        """
        command_names = {
            "C": "Create",
            "U": "Update",
            "D": "Delete",
            "E": "Evaluate"
        }
        
        entity_names = {
            "NODE": "node",
            "LINK": "edge" if entity_type == "LINK" else "link"
        }
        
        cmd_name = command_names.get(command_type, command_type)
        entity_name = entity_names.get(entity_type, entity_type.lower())
        
        # Special handling for link endpoint extraction
        if context and context.get("parsing_mode") == "extract_link_endpoints":
            return f"""
            Extract the source node ID and target node ID from this link creation instruction:

            "{instruction}"

            Return only Python code that sets result = {{"source": "source_node_id", "target": "target_node_id"}}
            """
            
        # Regular prompts for normal operations
        prompt = f"Generate Python code to {cmd_name.lower()} a knowledge graph {entity_name} based on this instruction:\n\n"
        prompt += f"\"{instruction}\"\n\n"
        
        # Add context for link creation
        if entity_type == "LINK" and command_type == "C":
            prompt += "The source_uid and target_uid are ALREADY SET. DO NOT try to set them. Focus ONLY on determining the relation type.\n\n"
            
            if context:
                prompt += "Additional context:\n"
                if "source_type" in context:
                    prompt += f"- Source node type: {context['source_type']}\n"
                if "target_type" in context:
                    prompt += f"- Target node type: {context['target_type']}\n"
        
        # Add other context if provided
        elif context and entity_type != "LINK":
            prompt += "Additional context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
                
        prompt += f"\nCreate code to {cmd_name.lower()} the {entity_name} according to the instruction. "
        prompt += f"Return only Python code that can be executed directly."
        
        return prompt
    
    def _prompt_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send prompts to the LLM and get a response.
        
        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The user prompt for the LLM
            
        Returns:
            LLM response string
            
        Raises:
            CodeGenerationError: If LLM prompting fails
        """
        try:
            from integration.net.ollama.ollama_api import prompt_model
            
            response = prompt_model(
                message=user_prompt,
                model=self.model_id,
                system_prompt=system_prompt
            )
            
            return response
        except Exception as e:
            raise CodeGenerationError(f"Failed to prompt LLM: {str(e)}")
    
    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from the LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            Extracted Python code string
        """
        import re
        
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
        return response.strip()
