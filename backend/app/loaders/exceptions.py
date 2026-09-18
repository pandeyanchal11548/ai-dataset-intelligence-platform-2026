"""Custom exceptions for the dataset loading pipeline."""


class DatasetLoaderError(Exception):
    """Base class for all dataset loader errors."""


class UnsupportedFileTypeError(DatasetLoaderError):
    """Raised when a file's extension is not CSV/Excel-family."""


class FileValidationError(DatasetLoaderError):
    """Raised when a file fails structural validation (empty, corrupt, unreadable)."""


class EmptyDatasetError(DatasetLoaderError):
    """Raised when a file parses but yields zero usable rows/columns."""
