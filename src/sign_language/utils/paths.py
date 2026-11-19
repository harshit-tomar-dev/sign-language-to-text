"""Path resolution and directory management utilities using pathlib."""

from pathlib import Path


def project_root() -> Path:
    """Return the absolute path to the project root directory.

    Searches upwards from this file until config.yaml or pyproject.toml is found,
    or falls back to two levels up from src/.

    Returns:
        Path: Project root directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    # Default fallback to 3 levels up from this file
    return current.parents[3]


def resolve(relative: str | Path) -> Path:
    """Resolve a relative path against the project root.

    Args:
        relative: Path string or Path object relative to project root.

    Returns:
        Path: Absolute resolved path.
    """
    rel_path = Path(relative)
    if rel_path.is_absolute():
        return rel_path
    return (project_root() / rel_path).resolve()


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating parents if necessary.

    Args:
        path: Path to create.

    Returns:
        Path: The same directory path guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
