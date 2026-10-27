"""Storage Tank Volume Calculator
Simple script to estimate the volume of a vertical cylindrical storage tank.
Computes volume in cubic meters, liters, and US barrels, with an optional fill percentage.
Formula: v = pi * r^2 * h
"""

import math

# Constants
PI = math.pi

class Tank:
    # Initialize tank with radius and height
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    # Calculate tank volume in cubic meters
    def calculate_volume(self):
        return PI * self.radius ** 2 * self.height

    # Calculate capacity in barrels (1 m³ ≈ 6.2898 barrels)
    def calculate_capacity(self):
        return self.calculate_volume() * 6.2898

def greet_operator():
    print("Welcome to the Storage Tank Volume Calculator!")


def main():
    greet_operator()

    # Get user inputs
    radius = float(input("Enter the radius of the tank: "))
    height = float(input("Enter the height of the tank: "))

    # Create a Circle object
    tank = Tank(radius, height)

    # Perform calculations
    volume=tank.calculate_volume() 
    capacity=tank.calculate_capacity()

    # Display results
    print(f"The volume of the tank is: ${volume:.2f} cubic meters")
    print(f"The capacity of the tank is: ${capacity:.2f} barrels")

if __name__ == "__main__":
    main()
