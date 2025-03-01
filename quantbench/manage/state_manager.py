from dataclasses import dataclass, field
from enum import Enum
from typing import List, Any, Optional


class ProcessStatus(Enum):
    """Process status enumeration for tracking the quantization workflow."""
    IDLE = "idle"
    CONVERTING = "converting"
    GENERATING_IMATRIX = "generating_imatrix"
    QUANTIZING = "quantizing"
    BENCHMARKING = "benchmarking"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class QuantizationState:
    """Data class to store the application state."""
    input_dir: str = ""
    output_dir: str = ""
    imatrix_option: str = ""
    output_format: str = "GGUF"
    quant_types: List[str] = field(default_factory=list)
    selected_quant_types: List[str] = field(default_factory=list)
    process_status: ProcessStatus = ProcessStatus.IDLE
    error_message: str = ""
    results_data: List[List[Any]] = field(default_factory=list)

    # Paths for important files
    base_model_path: Optional[str] = None
    imatrix_path: Optional[str] = None


class StateManager:
    """Singleton class for managing application state."""
    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = StateManager()
        return cls._instance

    def __init__(self):
        """Initialize the state manager."""
        if StateManager._instance is not None:
            raise Exception("StateManager is a singleton. Use get_instance() to access it.")
        self.state = QuantizationState()
        self.listeners = []

    def update_state(self, **kwargs):
        """Update state with the provided key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self._notify_listeners()

    def register_listener(self, listener):
        """Register a listener function that will be called when state changes."""
        if listener not in self.listeners:
            self.listeners.append(listener)

    def _notify_listeners(self):
        """Notify all registered listeners about state changes."""
        for listener in self.listeners:
            listener(self.state)
