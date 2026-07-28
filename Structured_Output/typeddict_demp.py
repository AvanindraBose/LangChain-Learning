from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    "name": "Avanindra",
    "age": "21"
}

#  Typed dict is only for telling the users about the schema. It will not enforce it,
#  i.e it will not give errors when the rules are violated

print(new_person)