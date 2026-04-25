"""
Exception classes for PyRLUtils.
"""

class InvalidRangeError(Exception):
    """
    Exception raised when a value is outside the allowed range.

    Attributes:
        message: Explanation of the error.
    """

    def __init__(self, message=None):
        """
        Initialize the exception.

        Args:
            message: Optional custom message. If None, uses default message.
        """
        self.message = "Invalid range error!" if message is None else message
        super().__init__(self.message)