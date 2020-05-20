"""
This module contains custom exceptions.
"""


class LanefinderException(Exception):
    """
    Generic exception for this project
    """
    pass


class PipeException(LanefinderException):
    """
    Exception caused by error during pipline execution
    """
    pass
