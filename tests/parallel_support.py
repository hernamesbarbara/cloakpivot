"""Parallel test execution utilities for CloakPivot test suite.

This module provides utilities to support pytest-xdist parallel execution,
including worker identification, environment setup, and resource isolation.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Callable

try:
    from pytest_xdist import is_xdist_worker
except ImportError:
    # Fallback if pytest-xdist is not available
    def is_xdist_worker() -> bool:
        """Fallback when pytest-xdist is not available."""
        return False


class ParallelTestSupport:
    """Utilities for parallel test execution with pytest-xdist."""

    @staticmethod
    def is_main_worker() -> bool:
        """Check if running in main process (not xdist worker)."""
        return not is_xdist_worker()

    @staticmethod
    def get_worker_count() -> int:
        """Get the number of parallel workers configured."""
        # Check various environment variables that might indicate worker count
        worker_count_vars = [
            'PYTEST_XDIST_WORKER_COUNT',
            'PYTEST_WORKERS',
            'PYTEST_NUM_WORKERS'
        ]

        for var in worker_count_vars:
            count_str = os.getenv(var)
            if count_str and count_str.isdigit():
                return int(count_str)

        # Default to 1 if not running in parallel
        return 1

    @staticmethod
    def get_worker_id() -> str:
        """Get current worker ID."""
        # pytest-xdist sets PYTEST_XDIST_WORKER environment variable
        worker_id = os.getenv('PYTEST_XDIST_WORKER')
        if worker_id:
            return worker_id

        # Alternative check for worker ID in process env
        if is_xdist_worker():
            # Try to extract from pytest worker input if available
            return f"worker_{os.getpid()}"

        return 'main'

    @staticmethod
    def setup_worker_environment(worker_id: str | None = None) -> dict[str, str]:
        """Set up worker-specific environment variables.

        Args:
            worker_id: Optional worker ID, auto-detected if not provided

        Returns:
            Dictionary of environment variables set for this worker
        """
        if worker_id is None:
            worker_id = ParallelTestSupport.get_worker_id()

        env_vars = {}

        # Worker-specific random seeds for reproducibility
        seed = hash(worker_id) & 0x7FFFFFFF  # Ensure positive 32-bit integer
        random.seed(seed)
        env_vars['PYTEST_WORKER_RANDOM_SEED'] = str(seed)

        # Store worker temp directory path but don't create it yet (lazy creation)
        temp_base = Path(tempfile.gettempdir()) / f'cloakpivot_worker_{worker_id}'
        worker_temp = str(temp_base)
        os.environ['WORKER_TEMP_DIR'] = worker_temp
        env_vars['WORKER_TEMP_DIR'] = worker_temp

        # Worker identification
        os.environ['PYTEST_CURRENT_WORKER'] = worker_id
        env_vars['PYTEST_CURRENT_WORKER'] = worker_id

        return env_vars

    @staticmethod
    def get_worker_temp_dir() -> Path:
        """Get the temporary directory for the current worker."""
        worker_temp = os.getenv('WORKER_TEMP_DIR')
        if worker_temp:
            temp_path = Path(worker_temp)
            # Create directory on first access (lazy creation)
            temp_path.mkdir(exist_ok=True)
            return temp_path

        # Fallback: create worker-specific temp dir
        worker_id = ParallelTestSupport.get_worker_id()
        temp_base = Path(tempfile.gettempdir()) / f'cloakpivot_worker_{worker_id}'
        temp_base.mkdir(exist_ok=True)
        return temp_base

    @staticmethod
    def cleanup_worker_environment() -> None:
        """Clean up worker-specific environment and temporary files."""
        worker_temp = os.getenv('WORKER_TEMP_DIR')
        if worker_temp:
            try:
                import shutil
                temp_path = Path(worker_temp)
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
            except (ImportError, OSError):
                # Best effort cleanup - don't fail tests if cleanup fails
                pass

    @staticmethod
    def is_parallel_execution() -> bool:
        """Check if tests are running in parallel mode."""
        return is_xdist_worker() or ParallelTestSupport.get_worker_count() > 1

    @staticmethod
    def get_optimal_worker_count() -> int:
        """Calculate optimal worker count based on system resources."""
        import multiprocessing

        # Respect explicit configuration
        explicit_workers = os.getenv('PYTEST_WORKERS')
        if explicit_workers and explicit_workers.isdigit():
            return int(explicit_workers)

        cpu_count = multiprocessing.cpu_count()

        # Check for resource limits
        max_workers_str = os.getenv('PYTEST_MAX_WORKERS', '8')
        min_workers_str = os.getenv('PYTEST_MIN_WORKERS', '1')

        try:
            max_workers = int(max_workers_str)
            min_workers = int(min_workers_str)
        except ValueError:
            max_workers = 8
            min_workers = 1

        # Conservative defaults based on system resources
        if cpu_count <= 2:
            optimal = 1  # Single-threaded on limited systems
        elif cpu_count <= 4:
            optimal = 2  # Half cores on small systems
        elif cpu_count <= 8:
            optimal = min(4, cpu_count - 1)  # Leave one core free
        else:
            optimal = min(max_workers, cpu_count // 2)  # Cap at max_workers, use half cores

        # Apply bounds
        return max(min_workers, min(optimal, max_workers))


class WorkerResourceManager:
    """Manages resources across parallel test workers."""

    def __init__(self, worker_id: str | None = None):
        """Initialize resource manager for a specific worker.

        Args:
            worker_id: Worker ID, auto-detected if not provided
        """
        self.worker_id = worker_id or ParallelTestSupport.get_worker_id()
        self._temp_dirs: list[Path] = []
        self._cleanup_callbacks: list[Callable[[], None]] = []

    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a worker-specific temporary directory.

        Args:
            prefix: Directory name prefix

        Returns:
            Path to created temporary directory
        """
        base_temp = ParallelTestSupport.get_worker_temp_dir()
        temp_dir = base_temp / f"{prefix}{len(self._temp_dirs)}"
        temp_dir.mkdir(exist_ok=True)

        self._temp_dirs.append(temp_dir)
        return temp_dir

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback to run when worker finishes.

        Args:
            callback: Function to call during cleanup
        """
        self._cleanup_callbacks.append(callback)

    def cleanup(self) -> None:
        """Clean up all resources managed by this worker."""
        # Run cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception:
                # Best effort cleanup - don't fail tests
                pass

        # Remove temporary directories
        import shutil
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                # Best effort cleanup
                pass

        self._temp_dirs.clear()
        self._cleanup_callbacks.clear()


# Global resource manager instance per worker
_worker_resource_manager: WorkerResourceManager | None = None


def get_worker_resource_manager() -> WorkerResourceManager:
    """Get the global resource manager for the current worker."""
    global _worker_resource_manager
    if _worker_resource_manager is None:
        _worker_resource_manager = WorkerResourceManager()
    return _worker_resource_manager


def setup_parallel_test_environment() -> None:
    """Set up the parallel test environment for the current worker."""
    # Initialize worker environment
    ParallelTestSupport.setup_worker_environment()

    # Initialize resource manager
    get_worker_resource_manager()


def teardown_parallel_test_environment() -> None:
    """Tear down the parallel test environment for the current worker."""
    global _worker_resource_manager

    # Clean up resource manager
    if _worker_resource_manager is not None:
        _worker_resource_manager.cleanup()
        _worker_resource_manager = None

    # Clean up worker environment
    ParallelTestSupport.cleanup_worker_environment()
