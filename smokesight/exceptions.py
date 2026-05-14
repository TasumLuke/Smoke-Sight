class SmokeSightError(Exception):
    """Base SmokeSight exception."""

class SmokeSightCalibrationError(SmokeSightError):
    """Raised when calibration fails."""

class SmokeSightRetrievalError(SmokeSightError):
    """Raised when retrieval fails."""
