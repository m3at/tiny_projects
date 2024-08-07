from ninja import ModelSchema, NinjaAPI, Schema

# from tasks.models import DummyTask, DummyTaskIn, DummyTaskOut
from tasks.models import DummyTask

api = NinjaAPI()


class DummyTaskIn(Schema):
    name: str | None = None


class DummyTaskOut(ModelSchema):
    class Meta:
        model = DummyTask
        fields = ["id", "date", "name"]
        # exclude = ['password', 'last_login', 'user_permissions']
        # fields_optional = ['id', 'date', 'name']


# @api.post("/hello")
# def hello(request, data: WildSchema):
#     return f"Hello world, {data.surroundings}"


# @api.post("/work/{task_id}")
@api.post("/add_task")
async def add_task(request, task_in: DummyTaskIn):
    # print(task_in.dict())
    # task = DummyTask.objects.create(**task_in.dict())
    task = await DummyTask.objects.acreate(**task_in.dict())
    return f"Created task: {task.name=} {task.date=} {task.id=}"


@api.get("/list_tasks", response=list[DummyTaskOut])
def list_tasks(request):
    qs = DummyTask.objects.all()
    return qs
