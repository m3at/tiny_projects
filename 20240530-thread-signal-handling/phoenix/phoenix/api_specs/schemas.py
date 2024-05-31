from ninja import Schema


class SuccessMessage(Schema):
    message: str


class FailMessage(Schema):
    message: str


class BirdCount(Schema):
    total_birds: int
