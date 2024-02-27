import torch
import math
import csv
import random

def check_cross(segment1, segment2):
    return False  # Check if two line segments cross each other
def generate_lines(num_points, boundary=1.5):  # generate a random convex polygon.
    angles_lengths = [(random.uniform(0, 2*math.pi), random.uniform(0.15, boundary)) for _ in range(num_points)]
    angles_lengths.sort()  # Sort the points by angle

    points = [torch.tensor([math.cos(angle) * length, math.sin(angle) * length]) for angle, length in angles_lengths]
    return points

def create_segments_from_points(points): # create line segments from the points
    segments = []
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  # Connect the last point to the first point
        segments.append((start_point, end_point))
    return segments

def save_to_csv(segments, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for segment in segments:
            writer.writerow([segment[0][0].item(), segment[0][1].item(), segment[1][0].item(), segment[1][1].item()])

num_segments = 40
points = generate_lines(num_segments)
segments = create_segments_from_points(points)
save_to_csv(segments, 'coordinates.csv')
