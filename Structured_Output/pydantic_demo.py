from pydantic import BaseModel,Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10,default=5, description="This will fetch the cgpa pf the studemt")

new_student = {'name':'Avanindra','age':23,'cgpa':8.98}

student = Student(**new_student)

print(student)