class InvalidActionError(Exception):
    """
    Exception raised for actions taken when the game is over

    """
    def __init__(self, message="An illegal move was attempted"):
        self.message = message
        super().__init__(self.message)