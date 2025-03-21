import asyncio
from enum import Enum
from typing import Any, Dict, Optional, Callable

from pydantic import field_validator

# Import the updated base models from the Knowledge Graph module.
from knowledge.graph.kg_models import KGNode, KGEdge


# ------------------------------------------------------------------------------
# Function Metadata Keys as Enums
# ------------------------------------------------------------------------------
class FunctionMetaKey(str, Enum):
    CODE = "code"
    CALLABLE = "callable"


# ------------------------------------------------------------------------------
# Recursive Reasoning Framework Models
# ------------------------------------------------------------------------------
class DataNode(KGNode):
    """
    DataNode is the basic unit in the Recursive Reasoning Framework.

    In addition to the basic KGNode properties (including creation and update timestamps),
    it adds:
      - `content`: in-memory data (of any type).
    """
    content: Optional[Any] = None

    def _parse_instructions(self, instructions: Any) -> callable:
        """
        TODO
        """
        return

    def evaluate(self, instructions: Any = None) -> Any:
        """
        Evaluate the node based on provided instructions.
        Otherwise, the current content is returned.

        Supported instruction types:
        - 'args' & 'kwargs' dict entries.
        - Natural language instructions.
        """
        parsed_instructions = self._parse_instructions(instructions)
        # TODO
        return self.content

    def update(self, instructions: Any) -> None:
        """
        Update the in-memory content.
        """
        parsed_instructions = self._parse_instructions(instructions)
        # TODO


class FunctionNode(DataNode):
    """
    FunctionNode represents any callable operation (including observations).

    It builds on DataNode by requiring that a callable (either provided directly or
    as a code string) be stored in its metadata under a key defined in FunctionMetaKey.
    When called, it executes that callable (synchronously or asynchronously) and updates
    its content with the result.
    """

    def evaluate(self, instructions: Any = None) -> Any:
        """
        Evaluate the function node based on natural language instructions.
        Instructions can be provided as:
          - A dict with keys 'args' and 'kwargs'
          - A string that is naively split into arguments
          - A complex prompt to be translated to code
        The underlying callable is executed and its result is stored in the content.
        """
        args = []
        kwargs = {}
        if isinstance(instructions, dict):
            args = instructions.get("args", [])
            kwargs = instructions.get("kwargs", {})
        elif isinstance(instructions, str):
            # Naively split the string into arguments (improvement: integrate with a proper NLP parser)
            args = instructions.split()

        if self.is_async:
            try:
                loop = asyncio.get_running_loop()
                task = asyncio.create_task(self._acall(*args, **kwargs))
                # Return the task; caller is responsible for awaiting it.
                return task
            except RuntimeError:
                # No event loop is running; execute the async call synchronously.
                return asyncio.run(self._acall(*args, **kwargs))
        else:
            return self._call(*args, **kwargs)

    # TODO This is problematic. Callables cannot be serialized...
    def _get_callable(self) -> Callable:
        """
        Retrieve the callable to execute.

        First checks for a direct callable reference (key 'callable'); if not found,
        it looks for a code string (key 'code') and executes it to retrieve a callable named `func`.
        """
        meta_callable = self.meta_props.get(FunctionMetaKey.CALLABLE.value)
        if meta_callable is not None and callable(meta_callable):
            return meta_callable

        code_str = self.meta_props.get(FunctionMetaKey.CODE.value)
        if not code_str:
            raise ValueError(
                "FunctionNode metadata must include either a callable reference or a code string "
                "with a callable definition (key: 'code')."
            )
        local_namespace: Dict[str, Any] = {}
        exec(code_str, {}, local_namespace)
        func = local_namespace.get("func")
        if not callable(func):
            raise ValueError("The provided code does not define a callable 'func'.")
        return func

    def _call(self, *args, **kwargs) -> Any:
        """
        Synchronously execute the function.

        If the function is asynchronous and an event loop is running,
        a task is created (which the caller should await). Otherwise,
        asyncio.run is used. The result is stored in the node's content.
        """
        func = self._get_callable()
        if asyncio.iscoroutinefunction(func):
            try:
                result = asyncio.create_task(func(*args, **kwargs))
            except RuntimeError:
                result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        self.content = result
        return result

    async def _acall(self, *args, **kwargs) -> Any:
        """
        Asynchronously execute the function.

        If the function is asynchronous, await it directly;
        if synchronous, execute it in a thread pool.
        The result is stored in the node's content.
        """
        func = self._get_callable()
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, func, *args, **kwargs)
        self.content = result
        return result


class OutcomeNode(FunctionNode):
    """
    OutcomeNode represents the outcome of a function node evaluation, with an associated weight.
    It maintains a target evaluation state, tracks the last evaluation state, and adjusts its weight
    based on the comparison between the actual result and the target state.
    """
    weight: float = 0.0
    target_eval_state: Any
    last_eval_state: Any = None

    @field_validator("weight")
    def validate_weight(cls, v: float) -> float:
        if v < -1.0 or v > 1.0:
            raise ValueError("Weight must be between -1.0 and 1.0.")
        return v

    def evaluate(self, instructions: Any = None) -> Any:
        """
        Evaluate the OutcomeNode by executing the underlying function and comparing the result
        to the target evaluation state. The last evaluation state is updated and the weight is
        adjusted (positive if the result matches the target, negative otherwise).
        Natural language instructions are supported in the same way as in FunctionNode.
        """
        result = super().evaluate(instructions)
        self.last_eval_state = result
        # Simple evaluation: if the result equals the target evaluation state, set positive weight.
        self.weight = 1.0 if result == self.target_eval_state else -1.0
        return result


# ------------------------------------------------------------------------------
# Strict Edge Definitions for Recursive Reasoning
# ------------------------------------------------------------------------------
class LinkRelation(str, Enum):
    """
    Enumeration of allowed edge relations.

    - NON_FUNCTIONAL: Any type of non-functional Node relationship.
    - EVAL_SEQUENCE: Chained evaluation Nodes. Evaluating any Node in the sequence will automatically trigger succeeding Node evaluation.
    - PARAMETER: Connects Function Nodes, propagating evaluation results as next Node evaluation parameters.
    - HIERARCHY: A parent–child (including historical) or producer-product relationship.
    - CASUAL: Node is created automatically in reaction to another Node (Event-Reaction-Action-Reaction-...)
    """
    NON_FUNCTIONAL = "non_functional"
    EVAL_SEQUENCE = "eval_sequence"
    PARAMETER = "parameter"
    HIERARCHY = "hierarchy"
    CASUAL = "casual"


class Link(KGEdge):
    """
    ReasoningEdge extends the base KGEdge to enforce strict relationships in the framework.

    The `relation` field must be one of the allowed types from the LinkRelation enum.
    """
    relation: LinkRelation

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(**kwargs)


# ------------------------------------------------------------------------------
# Meta-Nodes: Action & Event Nodes (Read-Only)
# ------------------------------------------------------------------------------
class ActionMetaNode(KGNode):
    """
    ActionMetaNode represents a record of a decision or reasoning step (e.g. by an LLM).

    It is read-only and stores metadata such as decision details, timestamp, and context.
    """
    # model_config = {"frozen": True}


class EventMetaNode(KGNode):
    """
    EventMetaNode represents external or asynchronous events that the graph can react to.

    It is read-only and stores metadata such as timing, source, and event-specific details.
    """
    # model_config = {"frozen": True}
