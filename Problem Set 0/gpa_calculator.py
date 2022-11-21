from typing import List
from college import Student, Course
import utils


def calculate_gpa(student: Student, courses: List[Course]) -> float:
    '''
    This function takes a student and a list of course
    It should compute the GPA for the student
    The GPA is the sum(hours of course * grade in course) / sum(hours of course)
    The grades come in the form: 'A+', 'A' and so on.
    But you can convert the grades to points using a static method in the course class
    To know how to use the Student and Course classes, see the file "college.py"  
    '''
    # TODO: ADD YOUR CODE HERE
    # utils.NotImplemented()
    # print(student.name+": "+student.id)
    # for course in courses:
    #     print(course.grades[student.id] + ": " + course.name)
    courses_dict = {}
    gpa = 0
    total_hours = 0
    for course in courses:
        if student.id in course.grades.keys():
            courses_dict[course] = (course.grades[student.id], course.hours)

    if len(courses_dict) != 0:
        for course in courses_dict:
            gpa += course.convert_grade_to_points(course.grades[student.id]) * course.hours
            total_hours += course.hours
        gpa = gpa / total_hours
    return gpa
