import uuid

from django.db import models
# from ninja import Schema, ModelSchema


# class Question(models.Model):
#     question_text = models.CharField(max_length=200)
#     pub_date = models.DateTimeField("date published")
#
#
# class Choice(models.Model):
#     question = models.ForeignKey(Question, on_delete=models.CASCADE)
#     choice_text = models.CharField(max_length=200)
#     votes = models.IntegerField(default=0)


# class WildSchema(Schema):
#     surroundings: str = "world"


class DummyTask(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=255, null=True)


# class TaskSchema(ModelSchema):
#     class Meta:
#         model = Task
#         # fields = ['id', 'date', 'name']
#         # exclude = ['password', 'last_login', 'user_permissions']
#         fields_optional = ['id', 'date', 'name']
