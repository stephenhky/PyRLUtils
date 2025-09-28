
class InvalidRangeError(Exception):
    def __init__(self, message=None):
        self.message = "Invalid range error!" if message is None else message
        super().__init__(self.message)
