#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 1
#
"""
1. Create a list with values 1–20.
2. Use a list comprehension to square all odd values in this list.
3. Request 4 numbers and sort the numbers in ascending order.
"""

print("Introduction to Python programming")

my_list = list(range(1,21))
print(my_list)

my_list = [x**2 if x%2 != 0 else x for x in my_list]
print(my_list)

input_list = list()
for i in range(4):
    input_list.append(input("Enter a number: "))
print("Input :", input_list)
print("Sorted:", sorted(input_list))

#
# Task 2
#

"""
1. Square all elements of a list.
2. Recursively calculate the sum of all elements in a list.
3.
"""
def square_list(input_list):
    return [x**2 for x in input_list]

def sum_recursive(input_list, i=0):
    if i == len(input_list)-1:
        return input_list[i]
    else:
        return input_list[i] + sum_recursive(input_list, i+1)

def compute_mean(input_list):
    return sum(input_list) / len(input_list)

input_list = list(range(1,6))
print("Input:  ", input_list)
print("Squared:", square_list(input_list))
print("Summed: ", sum_recursive(input_list))
print("Mean:   ", compute_mean(input_list))

#
# Task 3
#
"""
Class with
1. Constructor
2. Length function to compute eucledian distance
3. Add function to add two Vec2
4. Variable id and global class variable gid
"""
import math

class Vec2:

    gid = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = Vec2.gid
        Vec2.gid = Vec2.gid + 1
        print("id:", self.id, "current gid:", Vec2.gid)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def add(self, rhs):
        return Vec2(self.x+rhs.x, self.y+rhs.y)

# Demonstration example.
a = Vec2(1, 1)
b = Vec2(2, 2)
print("Vector a: ", a.x, a.y)
print("Vector b: ", b.x, b.y)
print("Length of a:", a.length())
c = a.add(b)
print("Vector c from a + b: ", c.x, c.y)
