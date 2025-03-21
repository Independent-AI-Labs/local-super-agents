"""
KGML Instruction Parser - Refactored to use the code generator interface.

This module parses natural language instructions in KGML commands and
converts them to executable Python code using a code generator.
"""

import ast
import logging
import re
from typing import Dict, Any, Optional

from knowledge.reasoning.dsl.kgml_code_generator_interface import (
    CodeGenerator, LLMCodeGenerator, CodeGenerationError
)


class KGMLParseError(Exception):
    """Exception raised for errors in parsing KGML instructions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class KGMLInstructionParser:
    """
    Parser for natural language instructions in KGML commands.
    
    Converts natural language instructions to secure Python code for execution,
    using a pluggable code generator.
    """

    def __init__(self, code_generator: Optional[CodeGenerator] = None, model_id: str = "qwen2.5-coder:14b"):
        """
        Initialize the parser with a code generator.
        
        Args:
            code_generator: The code generator to use
            model_id: Model ID for the default code generator if none provided
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Use provided code generator or create a default one
        self.code_generator = code_generator or LLMCodeGenerator(model_id=model_id)

    def parse_natural_language_instruction(self, 
                                          instruction: str,
                                          entity_type: str, 
                                          command_type: str,
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
            
        Raises:
            KGMLParseError: If parsing fails
        """
        self.logger.info(f"Parsing instruction for {command_type}► {entity_type}: {instruction}")

        try:
            # Generate code using the code generator
            code = self.code_generator.generate_code(
                instruction=instruction,
                entity_type=entity_type,
                command_type=command_type,
                context=context
            )
            
            return {
                "code": code,
                "original_instruction": instruction,
                "command_type": command_type,
                "entity_type": entity_type
            }
        except CodeGenerationError as e:
            self.logger.error(f"Error generating code: {str(e)}")
            raise KGMLParseError(f"Failed to generate code: {str(e)}", e.details)
        except Exception as e:
            self.logger.error(f"Unexpected error in instruction parsing: {str(e)}")
            raise KGMLParseError(f"Failed to parse instruction: {str(e)}")


class RuleBasedCodeGenerator(CodeGenerator):
    """
    Code generator that uses predefined rules and templates.
    
    This generator can be used for simple, well-defined tasks where
    LLM generation might be overkill or when LLM access is not available.
    """
    
    def __init__(self):
        """Initialize the rule-based code generator."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = self._load_templates()
        
    def generate_code(self, 
                     instruction: str, 
                     entity_type: str, 
                     command_type: str, 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate Python code using predefined rules and templates.
        
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
            # Get the appropriate template
            template_key = f"{entity_type.lower()}_{command_type.lower()}"
            template = self.templates.get(template_key)
            
            if not template:
                raise CodeGenerationError(f"No template found for {template_key}")
                
            # Extract parameters from the instruction
            params = self._extract_parameters(instruction, entity_type, command_type, context)
            
            # Fill in the template with the extracted parameters
            code = template.format(**params)
            
            # Validate the generated code
            self.validate_code(code)
            
            return code
        except Exception as e:
            raise CodeGenerationError(f"Failed to generate code: {str(e)}")
    
    def validate_code(self, code: str) -> None:
        """
        Validate generated code for syntax correctness.
        
        Args:
            code: The code to validate
            
        Raises:
            CodeGenerationError: If the code is invalid
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise CodeGenerationError(f"Invalid Python syntax: {e}")
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Load code templates for different entity types and commands.
        
        Returns:
            Dictionary of templates keyed by entity_type_command_type
        """
        return {
            # Node creation template
            "node_c": """
# Create a node based on the instruction
import datetime

# Set basic properties
node.type = "{node_type}"
node.meta_props.update({meta_props})

# Set additional properties
{additional_code}

# Create result
result = {{
    "status": "created",
    "node_id": node.uid,
    "node_type": node.type
}}
""",
            # Node update template
            "node_u": """
# Update a node based on the instruction
import datetime

# Update meta properties
{update_props}

# Set additional properties
{additional_code}

# Create result
result = {{
    "status": "updated",
    "node_id": node.uid
}}
""",
            # Node evaluation template
            "node_e": """
# Evaluate a node based on the instruction
{eval_code}

# Create result
result = {{
    {result_definition}
}}
""",
            # Link creation template
            "link_c": """
# Create a link based on the instruction
import datetime

# Set relation type
edge.relation = "{relation_type}"

# Set meta properties
edge.meta_props.update({meta_props})

# Create result
result = {{
    "status": "created", 
    "edge": f"{{edge.source_uid}}->{{edge.target_uid}}",
    "relation": edge.relation
}}
""",
            # Link update template
            "link_u": """
# Update a link based on the instruction
import datetime

# Update relation if specified
{relation_update}

# Update meta properties
{meta_update}

# Create result
result = {{
    "status": "updated",
    "edge": f"{{edge.source_uid}}->{{edge.target_uid}}"
}}
""",
            # Link extraction template
            "link_e": """
# Extract source and target from the instruction
instruction = "{instruction}"

{extraction_code}

# Set the result
result = {{
    "source": source,
    "target": target
}}
"""
        }
    
    def _extract_parameters(self, 
                           instruction: str, 
                           entity_type: str, 
                           command_type: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract parameters from the instruction for template filling.
        
        Args:
            instruction: The natural language instruction
            entity_type: The type of entity (NODE, LINK)
            command_type: The type of command (C, U, D, E)
            context: Optional additional context information
            
        Returns:
            Dictionary of parameters for template formatting
        """
        # This would need a much more sophisticated implementation for a real rule-based generator
        # This is just a simple example
        
        params = {
            "instruction": instruction,
            "meta_props": "{}",
            "additional_code": "# No additional properties to set",
            "update_props": "# No properties to update",
            "eval_code": "# Simple evaluation\neval_result = True",
            "result_definition": '"result": True',
            "relation_type": "GENERIC",
            "relation_update": "# No relation update",
            "meta_update": "# No meta properties to update",
            "extraction_code": """
# Simple extraction based on common patterns
source = ""
target = ""

# Try to find "between X and Y" pattern
if "between" in instruction.lower() and " and " in instruction.lower():
    parts = instruction.lower().split("between", 1)[1]
    if " and " in parts:
        entities = parts.split(" and ", 1)
        source = entities[0].strip()
        target = entities[1].strip().split(" ", 1)[0]
"""
        }
        
        # Extract node type from context or instruction
        if entity_type == "NODE" and command_type == "C":
            # Try to extract node type from instruction
            node_type_match = re.search(r'type(?:\s+is)?\s+(\w+)', instruction, re.IGNORECASE)
            if node_type_match:
                params["node_type"] = node_type_match.group(1)
            else:
                params["node_type"] = "GenericNode"
                
        # Extract relation type for links
        if entity_type == "LINK" and command_type == "C":
            # Try to extract relation type from instruction
            relation_match = re.search(r'relation(?:\s+is)?\s+(\w+)', instruction, re.IGNORECASE)
            if relation_match:
                params["relation_type"] = relation_match.group(1)
            elif "reply" in instruction.lower():
                params["relation_type"] = "REPLY_TO"
            elif "part of" in instruction.lower():
                params["relation_type"] = "PART_OF"
            elif "derived from" in instruction.lower() or "derives from" in instruction.lower():
                params["relation_type"] = "DERIVES_FROM"
            elif "augment" in instruction.lower():
                params["relation_type"] = "AUGMENTS"
            elif "reference" in instruction.lower():
                params["relation_type"] = "REFERENCES"
                
        return params
