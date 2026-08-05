from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

code = splitter.split_text('''
class Person:
    """Represents a generic person."""

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self):
        return f"My name is {self.name} and I am {self.age} years old."


class Student(Person):
    """Represents a student."""

    def __init__(self, name: str, age: int, roll_number: int):
        super().__init__(name, age)
        self.roll_number = roll_number
        self.marks = []

    def add_mark(self, mark: float):
        if mark < 0 or mark > 100:
            raise ValueError("Marks should be between 0 and 100.")
        self.marks.append(mark)

    def calculate_average(self):
        if not self.marks:
            return 0
        return sum(self.marks) / len(self.marks)

    def get_grade(self):
        average = self.calculate_average()

        if average >= 90:
            return "A"
        elif average >= 80:
            return "B"
        elif average >= 70:
            return "C"
        elif average >= 60:
            return "D"
        else:
            return "F"


class Teacher(Person):
    """Represents a teacher."""

    def __init__(self, name: str, age: int, subject: str):
        super().__init__(name, age)
        self.subject = subject

    def teach(self):
        return f"{self.name} is teaching {self.subject}."


class StudentManager:
    """Maintains a list of students."""

    def __init__(self):
        self.students = []

    def add_student(self, student: Student):
        self.students.append(student)

    def remove_student(self, roll_number: int):
        self.students = [
            student
            for student in self.students
            if student.roll_number != roll_number
        ]

    def find_student(self, roll_number: int):
        for student in self.students:
            if student.roll_number == roll_number:
                return student
        return None

    def display_students(self):
        for student in self.students:
            print(
                student.roll_number,
                student.name,
                student.calculate_average(),
                student.get_grade(),
            )


def calculate_class_average(manager: StudentManager):
    """Calculates the class average."""

    total = 0
    count = 0

    for student in manager.students:
        total += student.calculate_average()
        count += 1

    if count == 0:
        return 0

    return total / count


def generate_report(manager: StudentManager):
    """Generates a report."""

    print("=" * 40)
    print("CLASS REPORT")
    print("=" * 40)

    manager.display_students()

    print("-" * 40)
    print("Overall Average:", calculate_class_average(manager))


def main():
    manager = StudentManager()

    student1 = Student("Alice", 20, 101)
    student2 = Student("Bob", 21, 102)

    teacher = Teacher("Dr. Smith", 45, "Machine Learning")

    print(teacher.introduce())
    print(teacher.teach())

    for mark in [95, 91, 88]:
        student1.add_mark(mark)

    for mark in [78, 81, 84]:
        student2.add_mark(mark)

    manager.add_student(student1)
    manager.add_student(student2)

    generate_report(manager)


if __name__ == "__main__":
    main()
''')

print(len(code))

print(code[0])