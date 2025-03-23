"""
File tracking utility for VibeCheck.

This module extends the file modification tracking from cache_utils.py
with event-based notifications for various controllers.
"""

import os
import time
from typing import Dict, List, Callable, Set, Optional
import threading

from vibecheck.utils.cache_utils import FileModificationTracker
from vibecheck.utils.file_utils import list_files
from vibecheck import config


class FileTracker:
    """
    Event-based file tracking system that builds on FileModificationTracker.
    
    This class provides a way to register callbacks for file changes,
    which are triggered whenever files are modified, created, or deleted.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    _running = False
    _thread = None
    _callbacks: Dict[str, List[Callable]] = {}
    _pattern_callbacks: Dict[str, List[Callable]] = {}
    _project_path: Optional[str] = None
    _last_scan_time = 0
    
    def __new__(cls):
        """Singleton pattern to ensure only one tracker instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FileTracker, cls).__new__(cls)
                cls._callbacks = {}
                cls._pattern_callbacks = {}
                cls._running = False
            return cls._instance
    
    @classmethod
    def initialize(cls, project_path: str) -> None:
        """
        Initialize the file tracker for a project.
        
        Args:
            project_path: Path to the project to track
        """
        with cls._lock:
            cls._project_path = project_path
            
            # Start the tracker thread if it's not already running
            if not cls._running:
                cls._running = True
                cls._thread = threading.Thread(target=cls._tracker_thread, daemon=True)
                cls._thread.start()
    
    @classmethod
    def register_callback(cls, file_path: str, callback: Callable) -> None:
        """
        Register a callback for a specific file.
        
        Args:
            file_path: Path to the file to monitor
            callback: Function to call when the file changes
        """
        with cls._lock:
            # Normalize the path
            file_path = os.path.normpath(file_path)
            
            if file_path not in cls._callbacks:
                cls._callbacks[file_path] = []
            
            cls._callbacks[file_path].append(callback)
    
    @classmethod
    def register_pattern_callback(cls, pattern: str, callback: Callable) -> None:
        """
        Register a callback for files matching a pattern.
        
        Args:
            pattern: File pattern to match (e.g., "*.py", "architecture/*.json")
            callback: Function to call when matching files change
        """
        with cls._lock:
            if pattern not in cls._pattern_callbacks:
                cls._pattern_callbacks[pattern] = []
            
            cls._pattern_callbacks[pattern].append(callback)
    
    @classmethod
    def unregister_callback(cls, file_path: str, callback: Optional[Callable] = None) -> None:
        """
        Unregister a callback for a specific file.
        
        Args:
            file_path: Path to the file
            callback: Specific callback to remove, or None to remove all
        """
        with cls._lock:
            # Normalize the path
            file_path = os.path.normpath(file_path)
            
            if file_path in cls._callbacks:
                if callback is None:
                    # Remove all callbacks for this file
                    del cls._callbacks[file_path]
                else:
                    # Remove specific callback
                    try:
                        cls._callbacks[file_path].remove(callback)
                    except ValueError:
                        pass
    
    @classmethod
    def stop(cls) -> None:
        """Stop the tracker thread."""
        with cls._lock:
            cls._running = False
            if cls._thread:
                cls._thread.join(timeout=1.0)
                cls._thread = None
    
    @classmethod
    def _tracker_thread(cls) -> None:
        """Background thread that scans for file changes."""
        while cls._running:
            try:
                if cls._project_path:
                    cls._scan_for_changes()
            except Exception as e:
                print(f"Error in file tracker: {e}")
            
            # Sleep to prevent high CPU usage
            time.sleep(config.FILE_CHANGE_SCAN_INTERVAL)
    
    @classmethod
    def _scan_for_changes(cls) -> None:
        """Scan for file changes and trigger callbacks."""
        current_time = time.time()
        
        # Don't scan too frequently
        if current_time - cls._last_scan_time < config.FILE_CHANGE_SCAN_INTERVAL:
            return
        
        cls._last_scan_time = current_time
        
        # Get all files in the project
        all_files = list_files(
            cls._project_path,
            exclude_dirs=config.DEFAULT_EXCLUDE_DIRS + ['.vibecheck']
        )
        
        changed_files = []
        
        # Check each file for changes
        for file_path in all_files:
            abs_path = os.path.join(cls._project_path, file_path)
            
            if not os.path.exists(abs_path):
                continue
            
            # Use existing FileModificationTracker to check for changes
            has_changed = FileModificationTracker.has_file_changed(
                abs_path, 
                previous_mtime=cls._get_last_modified_time(file_path)
            )
            
            if has_changed:
                changed_files.append(file_path)
                cls._update_last_modified_time(file_path)
        
        # Trigger callbacks for changed files
        for file_path in changed_files:
            cls._trigger_callbacks(file_path)
    
    @classmethod
    def _trigger_callbacks(cls, file_path: str) -> None:
        """
        Trigger callbacks for a changed file.
        
        Args:
            file_path: Path to the changed file
        """
        with cls._lock:
            # Normalize the path
            norm_path = os.path.normpath(file_path)
            
            # Trigger specific file callbacks
            if norm_path in cls._callbacks:
                for callback in cls._callbacks[norm_path]:
                    try:
                        callback(file_path)
                    except Exception as e:
                        print(f"Error in file change callback: {e}")
            
            # Trigger pattern callbacks
            for pattern, callbacks in cls._pattern_callbacks.items():
                if cls._matches_pattern(file_path, pattern):
                    for callback in callbacks:
                        try:
                            callback(file_path)
                        except Exception as e:
                            print(f"Error in pattern callback: {e}")
    
    @classmethod
    def _get_last_modified_time(cls, file_path: str) -> Optional[float]:
        """
        Get the last known modification time for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Modification time or None if not known
        """
        # We can use FileModificationTracker's cache for this
        abs_path = os.path.join(cls._project_path, file_path)
        return FileModificationTracker.get_file_modification_time(abs_path)
    
    @classmethod
    def _update_last_modified_time(cls, file_path: str) -> None:
        """
        Update the last known modification time for a file.
        
        Args:
            file_path: Path to the file
        """
        # FileModificationTracker will automatically update its cache 
        # when has_file_changed is called, so we don't need to do anything here
        pass
    
    @classmethod
    def _matches_pattern(cls, file_path: str, pattern: str) -> bool:
        """
        Check if a file path matches a pattern.
        
        Args:
            file_path: Path to check
            pattern: Pattern to match against
            
        Returns:
            True if the file matches the pattern, False otherwise
        """
        # Simple pattern matching
        if pattern == "*":
            return True
        
        # Handle file extension patterns
        if pattern.startswith("*."):
            ext = pattern[1:]
            return file_path.endswith(ext)
        
        # Handle directory patterns
        if pattern.endswith("/*"):
            dir_pattern = pattern[:-1]
            return file_path.startswith(dir_pattern)
        
        # Handle combined patterns
        if "*" in pattern:
            parts = pattern.split("*")
            return file_path.startswith(parts[0]) and file_path.endswith(parts[1])
        
        # Exact match
        return file_path == pattern
