import datetime
import json
import logging
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, List, Callable, Optional, Tuple

import networkx as nx
from pydantic import BaseModel

# Set up logging (adjust as needed)
logging.basicConfig(level=logging.DEBUG)


###############################################################################
# Pydantic Models for Nodes and Edges with Provenance Fields
###############################################################################

class KGNode(BaseModel):
    uid: str
    type: Optional[str] = None
    meta_props: Dict[str, Any] = {}
    # Added audit trail / provenance fields
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class KGEdge(KGNode):
    source_uid: str
    target_uid: str


###############################################################################
# Persistence Backend Interfaces (unchanged)
###############################################################################

class PersistenceBackend(ABC):
    @abstractmethod
    def save(self, kg: "KnowledgeGraph") -> None:
        """Save the current graph to persistent storage."""
        pass

    @abstractmethod
    def load(self) -> "KnowledgeGraph":
        """Load and return a graph from persistent storage."""
        pass


class FilePersistenceBackend(PersistenceBackend):
    """A simple file-based persistence backend that uses JSON serialization."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, kg: "KnowledgeGraph") -> None:
        data = nx.node_link_data(kg._graph)  # produces data with key "links"
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
        kg.logger.debug(f"Graph saved to {self.file_path}")

    def load(self) -> "KnowledgeGraph":
        with open(self.file_path, "r") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, edges="links")
        kg = KnowledgeGraph()
        kg._graph = graph
        kg.logger.debug(f"Graph loaded from {self.file_path}")
        return kg


###############################################################################
# The KnowledgeGraph Class with Extended Versioning and Audit Trail
###############################################################################

class KnowledgeGraph:
    def __init__(self, persistence_backend: Optional[PersistenceBackend] = None):
        # Underlying NetworkX graph
        self._graph = nx.DiGraph()
        # Lock for thread safety
        self._lock = threading.RLock()
        # Dictionary to hold event hooks: event name -> list of callbacks
        self._hooks: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}
        self._persistence_backend = persistence_backend
        # Global version snapshots (for the whole graph)
        self._versions: Dict[int, Dict[str, Any]] = {}
        self._version_counter = 0
        # New: Per-node and per-edge version histories for fine-grained rollback.
        self._node_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._edge_versions: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        # Simple audit log list (could be persisted separately)
        self.audit_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    # --------------------------------------------------------------------------
    # Transaction and Concurrency
    # --------------------------------------------------------------------------
    @contextmanager
    def transaction(self):
        self.logger.debug("Acquiring transaction lock")
        self._lock.acquire()
        try:
            yield
            self.logger.debug("Transaction committed")
        except Exception as e:
            self.logger.error(f"Transaction error: {e}")
            raise
        finally:
            self._lock.release()
            self.logger.debug("Transaction lock released")

    # --------------------------------------------------------------------------
    # Event Hooks and Observability
    # --------------------------------------------------------------------------
    def register_hook(self, event: str, callback: Callable[[str, Dict[str, Any]], None]):
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
        self.logger.debug(f"Hook registered for event '{event}'")

    def _trigger_hook(self, event: str, data: Dict[str, Any]):
        for callback in self._hooks.get(event, []):
            try:
                callback(event, data)
            except Exception as e:
                self.logger.error(f"Error in hook for event '{event}': {e}")

    def _record_audit(self, event: str, details: Dict[str, Any]):
        # Append to an audit log list (could also be written to disk)
        entry = {"event": event, "timestamp": datetime.datetime.now().isoformat(), **details}
        self.audit_log.append(entry)
        self.logger.debug(f"Audit: {entry}")

    # --------------------------------------------------------------------------
    # Node Operations with Automatic Versioning and Timestamps
    # --------------------------------------------------------------------------
    def add_node(self, node: KGNode):
        with self._lock:
            now = datetime.datetime.now().isoformat()
            if not node.created_at:
                node.created_at = now
            node.updated_at = now

            if node.uid in self._graph:
                self.logger.warning(f"Node '{node.uid}' already exists. Overwriting.")
            self._graph.add_node(node.uid, **node.model_dump())
            self.logger.debug(f"Node added: {node.model_dump()}")

            # Initialize version history for the node if not present
            self._node_versions.setdefault(node.uid, []).append({
                "data": node.model_dump(),
                "timestamp": now,
                "message": "Created node"
            })
            self._trigger_hook("node_added", node.model_dump())
            self._record_audit("node_added", {"node_id": node.uid})

    def update_node(self, node_id: str, properties: Dict[str, Any]):
        with self._lock:
            if node_id not in self._graph:
                self.logger.error(f"Cannot update non-existent node: {node_id}")
                raise ValueError(f"Node {node_id} does not exist")
            now = datetime.datetime.now().isoformat()
            properties["updated_at"] = now
            self._graph.nodes[node_id].update(properties)
            self.logger.debug(f"Node updated: {node_id} with {properties}")
            self._trigger_hook("node_updated", {"id": node_id, "properties": properties})
            self._record_audit("node_updated", {"node_id": node_id, "properties": properties})
            # Save the new version snapshot
            current_data = self._graph.nodes[node_id].copy()
            self._node_versions.setdefault(node_id, []).append({
                "data": current_data,
                "timestamp": now,
                "message": "Updated node"
            })

    def get_node(self, node_id: str) -> Optional[KGNode]:
        with self._lock:
            if node_id in self._graph:
                data = self._graph.nodes[node_id]
                return KGNode(**data)
            return None

    def remove_node(self, node_id: str):
        with self._lock:
            if node_id in self._graph:
                self._graph.remove_node(node_id)
                self.logger.debug(f"Node removed: {node_id}")
                self._trigger_hook("node_removed", {"id": node_id})
                self._record_audit("node_removed", {"node_id": node_id})
            else:
                self.logger.warning(f"Attempted to remove non-existent node: {node_id}")

    # --------------------------------------------------------------------------
    # Edge Operations with Automatic Versioning and Timestamps
    # --------------------------------------------------------------------------
    def add_edge(self, edge: KGEdge):
        with self._lock:
            if not self._graph.has_node(edge.source_uid) or not self._graph.has_node(edge.target_uid):
                self.logger.error("Both nodes must exist before adding an edge.")
                raise ValueError("Both nodes must exist before adding an edge.")
            now = datetime.datetime.now().isoformat()
            if not edge.created_at:
                edge.created_at = now
            edge.updated_at = now
            self._graph.add_edge(edge.source_uid, edge.target_uid, **edge.model_dump())
            self.logger.debug(f"Edge added: {edge.model_dump()}")
            self._trigger_hook("edge_added", edge.model_dump())
            self._record_audit("edge_added", {"source": edge.source_uid, "target": edge.target_uid})
            # Save version snapshot for this edge
            key = (edge.source_uid, edge.target_uid)
            self._edge_versions.setdefault(key, []).append({
                "data": edge.model_dump(),
                "timestamp": now,
                "message": "Created edge"
            })

    def update_edge(self, source: str, target: str, properties: Dict[str, Any]):
        with self._lock:
            if not self._graph.has_edge(source, target):
                self.logger.error(f"Cannot update non-existent edge: {source} -> {target}")
                raise ValueError(f"Edge from {source} to {target} does not exist")
            now = datetime.datetime.now().isoformat()
            properties["updated_at"] = now
            self._graph.edges[source, target].update(properties)
            self.logger.debug(f"Edge updated: {source}->{target} with {properties}")
            self._trigger_hook("edge_updated", {"source": source, "target": target, "properties": properties})
            self._record_audit("edge_updated", {"source": source, "target": target, "properties": properties})
            # Save new version snapshot for the edge
            key = (source, target)
            current_data = self._graph.edges[source, target].copy()
            self._edge_versions.setdefault(key, []).append({
                "data": current_data,
                "timestamp": now,
                "message": "Updated edge"
            })

    def get_edge(self, source: str, target: str) -> Optional[KGEdge]:
        with self._lock:
            if self._graph.has_edge(source, target):
                data = self._graph.edges[source, target]
                return KGEdge(source=source, target=target, **data)
            return None

    def remove_edge(self, source: str, target: str):
        with self._lock:
            if self._graph.has_edge(source, target):
                self._graph.remove_edge(source, target)
                self.logger.debug(f"Edge removed: {source} -> {target}")
                self._trigger_hook("edge_removed", {"source": source, "target": target})
                self._record_audit("edge_removed", {"source": source, "target": target})
            else:
                self.logger.warning(f"Attempted to remove non-existent edge: {source} -> {target}")

    # --------------------------------------------------------------------------
    # Query / Indexing (unchanged)
    # --------------------------------------------------------------------------
    def query_nodes(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None, **properties) -> List[KGNode]:
        with self._lock:
            result = []
            for n, data in self._graph.nodes(data=True):
                if all(data.get(k) == v for k, v in properties.items()):
                    if filter_func is None or filter_func(data):
                        result.append(KGNode(**data))
            self.logger.debug(f"Query nodes with properties {properties} and filter_func {filter_func}: Found {len(result)} results")
            return result

    def query_edges(self, filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None, **properties) -> List[KGEdge]:
        with self._lock:
            result = []
            for u, v, data in self._graph.edges(data=True):
                if all(data.get(key) == value for key, value in properties.items()):
                    if filter_func is None or filter_func(data):
                        result.append(KGEdge(source=u, target=v, **data))
            self.logger.debug(f"Query edges with properties {properties} and filter_func {filter_func}: Found {len(result)} results")
            return result

    # --------------------------------------------------------------------------
    # Persistence / Versioning for the Entire Graph (Global Snapshot)
    # --------------------------------------------------------------------------
    def snapshot(self, message: Optional[str] = None) -> int:
        with self._lock:
            data = nx.node_link_data(self._graph, edges="links")
            timestamp = datetime.datetime.now().isoformat()
            self._version_counter += 1
            self._versions[self._version_counter] = {
                "data": data,
                "timestamp": timestamp,
                "message": message,
            }
            self.logger.debug(f"Global snapshot created for version {self._version_counter}")
            self._trigger_hook("snapshot", {"version": self._version_counter, "timestamp": timestamp, "message": message})
            self._record_audit("snapshot", {"version": self._version_counter})
            return self._version_counter

    def rollback(self, version: int):
        with self._lock:
            if version in self._versions:
                snapshot = self._versions[version]
                data = snapshot["data"]
                self._graph = nx.node_link_graph(data, edges="links")
                self.logger.debug(f"Global rollback to version {version}")
                self._trigger_hook("rollback", {"version": version})
                self._record_audit("rollback", {"version": version})
            else:
                self.logger.error(f"Version {version} not found for rollback.")
                raise ValueError(f"Version {version} not found.")

    def list_snapshots(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"version": ver, "timestamp": snap.get("timestamp"), "message": snap.get("message")}
                    for ver, snap in self._versions.items()]

    def save(self):
        if self._persistence_backend:
            with self._lock:
                self._persistence_backend.save(self)
                self.logger.debug("Graph saved via persistence backend")
                self._trigger_hook("save", {})
                self._record_audit("save", {})
        else:
            self.logger.error("No persistence backend provided.")
            raise RuntimeError("No persistence backend provided.")

    def load(self):
        if self._persistence_backend:
            loaded_kg = self._persistence_backend.load()
            with self._lock:
                self._graph = loaded_kg._graph
                self._versions = loaded_kg._versions
            self.logger.debug("Graph loaded via persistence backend")
            self._trigger_hook("load", {})
            self._record_audit("load", {})
        else:
            self.logger.error("No persistence backend provided.")
            raise RuntimeError("No persistence backend provided.")

    # --------------------------------------------------------------------------
    # Fine-grained Rollback Methods for Nodes and Edges
    # --------------------------------------------------------------------------
    def rollback_node(self, node_id: str, version_index: int):
        with self._lock:
            if node_id not in self._node_versions:
                self.logger.error(f"No version history for node {node_id}")
                raise ValueError(f"No version history for node {node_id}")
            versions = self._node_versions[node_id]
            if not (0 <= version_index < len(versions)):
                self.logger.error(f"Invalid version index {version_index} for node {node_id}")
                raise ValueError(f"Invalid version index for node {node_id}")
            snapshot = versions[version_index]["data"]
            self._graph.nodes[node_id].clear()
            self._graph.nodes[node_id].update(snapshot)
            self.logger.debug(f"Node {node_id} rolled back to version index {version_index}")
            self._trigger_hook("node_rolled_back", {"node_id": node_id, "version_index": version_index})
            self._record_audit("node_rolled_back", {"node_id": node_id, "version_index": version_index})

    def rollback_edge(self, source: str, target: str, version_index: int):
        with self._lock:
            key = (source, target)
            if key not in self._edge_versions:
                self.logger.error(f"No version history for edge {source} -> {target}")
                raise ValueError(f"No version history for edge {source} -> {target}")
            versions = self._edge_versions[key]
            if not (0 <= version_index < len(versions)):
                self.logger.error(f"Invalid version index {version_index} for edge {source} -> {target}")
                raise ValueError(f"Invalid version index for edge {source} -> {target}")
            snapshot = versions[version_index]["data"]
            self._graph.edges[source, target].clear()
            self._graph.edges[source, target].update(snapshot)
            self.logger.debug(f"Edge {source}->{target} rolled back to version index {version_index}")
            self._trigger_hook("edge_rolled_back", {"source": source, "target": target, "version_index": version_index})
            self._record_audit("edge_rolled_back", {"source": source, "target": target, "version_index": version_index})

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"nodes": self._graph.number_of_nodes(), "edges": self._graph.number_of_edges()}


###############################################################################
# Example Usage
###############################################################################

if __name__ == "__main__":
    # A simple hook to log events to standard output
    def log_event(event: str, data: Dict[str, Any]):
        print(f"Event: {event} | Data: {data}")


    # Initialize a KnowledgeGraph with a file-based persistence backend
    backend = FilePersistenceBackend("kg_graph.json")
    kg = KnowledgeGraph(persistence_backend=backend)

    # Register hooks for observability
    kg.register_hook("node_added", log_event)
    kg.register_hook("edge_added", log_event)
    kg.register_hook("snapshot", log_event)

    # Add nodes and an edge within a transaction
    with kg.transaction():
        node_a = KGNode(uid="A", type="Entity", properties={"name": "Alpha"})
        node_b = KGNode(uid="B", type="Entity", properties={"name": "Beta"})
        kg.add_node(node_a)
        kg.add_node(node_b)
        edge_ab = KGEdge(source_uid="A", target_uid="B", type="Relationship", properties={"relation": "connects"})
        kg.add_edge(edge_ab)

    # Create a global snapshot
    version = kg.snapshot(message="Initial snapshot after adding A, B and their edge")
    print(f"Global snapshot version: {version}")

    # Update a node's properties (automatically records a new version)
    kg.update_node("A", {"status": "active"})

    # Fine-grained rollback example: rollback node A to its initial version (index 0)
    kg.rollback_node("A", 0)
    node_a_after = kg.get_node("A")
    print(f"Node A after rollback: {node_a_after.model_dump()}")

    # Print graph statistics and audit log
    print("Graph stats:", kg.stats())
    print("Audit log entries:", kg.audit_log)
