"""
Cache utilities for VibeCheck.

This module provides utilities for caching analysis results and other data
to improve performance.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from vibecheck.config import CACHE_DIR


class AnalysisCache:
    """
    Cache for analysis results.
    
    This class provides methods for storing and retrieving analysis results
    from a cache to avoid redundant computations.
    """
    
    @staticmethod
    def get_cached_analysis(project_path: str, file_path: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis results for a file.

        Args:
            project_path (str): The path to the project
            file_path (str): The path to the file, relative to the project root
            analysis_type (str): The type of analysis (e.g., 'security', 'implementation')

        Returns:
            Optional[Dict[str, Any]]: The cached analysis results, or None if not found or expired
        """
        cache_path = AnalysisCache._get_cache_file_path(project_path, file_path, analysis_type)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if the cache is expired
            if AnalysisCache._is_cache_expired(cached_data):
                return None
            
            return cached_data.get('data')
        
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Error reading cache file {cache_path}: {e}")
            return None
    
    @staticmethod
    def cache_analysis(project_path: str, file_path: str, analysis_type: str, 
                      data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """
        Cache analysis results for a file.

        Args:
            project_path (str): The path to the project
            file_path (str): The path to the file, relative to the project root
            analysis_type (str): The type of analysis (e.g., 'security', 'implementation')
            data (Dict[str, Any]): The analysis results to cache
            ttl_seconds (int, optional): Time-to-live in seconds. Defaults to 3600 (1 hour).

        Returns:
            bool: True if the data was cached successfully, False otherwise
        """
        cache_path = AnalysisCache._get_cache_file_path(project_path, file_path, analysis_type)
        
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Prepare the cache data with metadata
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'expires': (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat(),
                'file_path': file_path,
                'analysis_type': analysis_type,
                'data': data
            }
            
            # Write the cache file
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error writing cache file {cache_path}: {e}")
            return False
    
    @staticmethod
    def invalidate_cache(project_path: str, file_path: Optional[str] = None, 
                        analysis_type: Optional[str] = None) -> bool:
        """
        Invalidate cached analysis results.

        Args:
            project_path (str): The path to the project
            file_path (Optional[str], optional): The path to the file, or None to invalidate all files. Defaults to None.
            analysis_type (Optional[str], optional): The type of analysis, or None to invalidate all types. Defaults to None.

        Returns:
            bool: True if the cache was invalidated successfully, False otherwise
        """
        try:
            cache_dir = os.path.join(project_path, CACHE_DIR)
            
            if not os.path.exists(cache_dir):
                return True  # Nothing to invalidate
            
            # If both file_path and analysis_type are None, delete the entire cache directory
            if file_path is None and analysis_type is None:
                import shutil
                shutil.rmtree(cache_dir)
                return True
            
            # If only analysis_type is specified, delete that analysis type directory
            if file_path is None and analysis_type is not None:
                analysis_dir = os.path.join(cache_dir, analysis_type)
                if os.path.exists(analysis_dir):
                    import shutil
                    shutil.rmtree(analysis_dir)
                return True
            
            # If only file_path is specified, delete all analysis types for that file
            if file_path is not None and analysis_type is None:
                normalized_path = file_path.replace('/', '_').replace('\\', '_')
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        if file.startswith(normalized_path):
                            os.remove(os.path.join(root, file))
                return True
            
            # If both file_path and analysis_type are specified, delete that specific cache file
            if file_path is not None and analysis_type is not None:
                cache_path = AnalysisCache._get_cache_file_path(project_path, file_path, analysis_type)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                return True
            
            return True
        
        except Exception as e:
            print(f"Error invalidating cache: {e}")
            return False
    
    @staticmethod
    def get_cache_status(project_path: str) -> Dict[str, Any]:
        """
        Get the status of the cache.

        Args:
            project_path (str): The path to the project

        Returns:
            Dict[str, Any]: A dictionary with cache status information
        """
        cache_dir = os.path.join(project_path, CACHE_DIR)
        
        if not os.path.exists(cache_dir):
            return {
                'exists': False,
                'size': 0,
                'entry_count': 0,
                'analysis_types': [],
                'created_at': None,
                'last_modified': None
            }
        
        # Get cache statistics
        total_size = 0
        entry_count = 0
        analysis_types = set()
        newest_time = 0
        oldest_time = float('inf')
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                total_size += file_size
                entry_count += 1
                
                # Get file modification times
                mtime = os.path.getmtime(file_path)
                newest_time = max(newest_time, mtime)
                oldest_time = min(oldest_time, mtime)
                
                # Get analysis type from directory name
                analysis_type = os.path.basename(os.path.dirname(file_path))
                analysis_types.add(analysis_type)
        
        return {
            'exists': True,
            'size': total_size,
            'size_human': AnalysisCache._format_size(total_size),
            'entry_count': entry_count,
            'analysis_types': list(analysis_types),
            'created_at': datetime.fromtimestamp(oldest_time).isoformat() if oldest_time != float('inf') else None,
            'last_modified': datetime.fromtimestamp(newest_time).isoformat() if newest_time > 0 else None
        }
    
    @staticmethod
    def cleanup_cache(project_path: str, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old cache entries.

        Args:
            project_path (str): The path to the project
            max_age_days (int, optional): Maximum age of cache entries in days. Defaults to 7.

        Returns:
            Dict[str, Any]: A dictionary with cleanup results
        """
        cache_dir = os.path.join(project_path, CACHE_DIR)
        
        if not os.path.exists(cache_dir):
            return {
                'cleaned': False,
                'deleted_entries': 0,
                'freed_space': 0
            }
        
        deleted_count = 0
        freed_space = 0
        max_age_seconds = max_age_days * 86400  # 86400 seconds = 1 day
        now = time.time()
        
        try:
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if the file is older than max_age_days
                    mtime = os.path.getmtime(file_path)
                    age_seconds = now - mtime
                    
                    if age_seconds > max_age_seconds:
                        # Get file size before deleting
                        file_size = os.path.getsize(file_path)
                        freed_space += file_size
                        
                        # Delete the file
                        os.remove(file_path)
                        deleted_count += 1
            
            # Remove empty directories
            for root, dirs, files in os.walk(cache_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
            
            return {
                'cleaned': True,
                'deleted_entries': deleted_count,
                'freed_space': freed_space,
                'freed_space_human': AnalysisCache._format_size(freed_space)
            }
        
        except Exception as e:
            print(f"Error cleaning up cache: {e}")
            return {
                'cleaned': False,
                'error': str(e),
                'deleted_entries': deleted_count,
                'freed_space': freed_space
            }
    
    @staticmethod
    def _get_cache_file_path(project_path: str, file_path: str, analysis_type: str) -> str:
        """
        Get the path to a cache file.

        Args:
            project_path (str): The path to the project
            file_path (str): The path to the file, relative to the project root
            analysis_type (str): The type of analysis

        Returns:
            str: The path to the cache file
        """
        # Normalize the file path for use in a filename
        normalized_path = file_path.replace('/', '_').replace('\\', '_')
        
        # Create the cache file path
        cache_dir = os.path.join(project_path, CACHE_DIR, analysis_type)
        return os.path.join(cache_dir, f"{normalized_path}.json")
    
    @staticmethod
    def _is_cache_expired(cache_data: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is expired.

        Args:
            cache_data (Dict[str, Any]): The cache data

        Returns:
            bool: True if the cache is expired, False otherwise
        """
        try:
            expires_str = cache_data.get('expires')
            if not expires_str:
                return True
            
            expires = datetime.fromisoformat(expires_str)
            now = datetime.now()
            
            return now > expires
        
        except Exception:
            return True  # If we can't parse the expiration time, assume it's expired
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        Format a size in bytes as a human-readable string.

        Args:
            size_bytes (int): Size in bytes

        Returns:
            str: Human-readable size
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.2f} TB"


class FileModificationTracker:
    """
    Tracker for file modifications.
    
    This class provides methods for tracking file modifications to determine
    when cache entries need to be invalidated.
    """
    
    @staticmethod
    def get_file_hash(file_path: str) -> Optional[str]:
        """
        Get a hash of a file's contents.

        Args:
            file_path (str): The path to the file

        Returns:
            Optional[str]: A hash of the file's contents, or None if the file could not be read
        """
        try:
            import hashlib
            
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                
                # Read the file in chunks to avoid loading large files into memory
                chunk_size = 8192
                while chunk := f.read(chunk_size):
                    file_hash.update(chunk)
                
                return file_hash.hexdigest()
        
        except Exception as e:
            print(f"Error calculating file hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def get_file_modification_time(file_path: str) -> Optional[float]:
        """
        Get the modification time of a file.

        Args:
            file_path (str): The path to the file

        Returns:
            Optional[float]: The modification time of the file, or None if the file could not be accessed
        """
        try:
            return os.path.getmtime(file_path)
        except Exception as e:
            print(f"Error getting modification time for {file_path}: {e}")
            return None
    
    @staticmethod
    def has_file_changed(file_path: str, previous_hash: Optional[str] = None,
                        previous_mtime: Optional[float] = None) -> bool:
        """
        Check if a file has changed.

        Args:
            file_path (str): The path to the file
            previous_hash (Optional[str], optional): The previous hash of the file. Defaults to None.
            previous_mtime (Optional[float], optional): The previous modification time of the file. Defaults to None.

        Returns:
            bool: True if the file has changed, False otherwise
        """
        # If we have a previous modification time, check that first (faster)
        if previous_mtime is not None:
            current_mtime = FileModificationTracker.get_file_modification_time(file_path)
            if current_mtime is None:
                return True  # Can't determine, assume changed
            
            if current_mtime > previous_mtime:
                return True  # File has been modified
            
            # If the modification time is the same, no need to check the hash
            return False
        
        # If we don't have a previous modification time but have a hash, check the hash
        if previous_hash is not None:
            current_hash = FileModificationTracker.get_file_hash(file_path)
            if current_hash is None:
                return True  # Can't determine, assume changed
            
            return current_hash != previous_hash
        
        # If we don't have either, assume the file has changed
        return True
