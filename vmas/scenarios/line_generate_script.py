import torch
import math
import csv

def generate_line_segments(num_segments):
    path_segments = []
    num_points = num_segments * 2

    for i in range(0, num_points, 2):
        if i < num_points // 3:
            start_point = torch.tensor([i * 3.0 / num_points, 0.0]) * 2 - 1
            end_point = torch.tensor([(i+1) * 3.0 / num_points, 0.0]) * 2 - 1
        elif i < 2 * num_points // 3:
            start_point = torch.tensor([1.0, (i - num_points // 3) * 3.0 / num_points]) * 2 - 1
            end_point = torch.tensor([1.0, ((i+1) - num_points // 3) * 3.0 / num_points]) * 2 - 1
        else:
            start_angle = (i - 2 * num_points // 3) * 2 * math.pi / (num_points // 3)
            end_angle = ((i+1) - 2 * num_points // 3) * 2 * math.pi / (num_points // 3)
            start_point = torch.tensor([math.cos(start_angle), math.sin(start_angle)]) * 0.5 + torch.tensor([0.5, 0.5])
            end_point = torch.tensor([math.cos(end_angle), math.sin(end_angle)]) * 0.5 + torch.tensor([0.5, 0.5])

        path_segments.append((start_point, end_point))

    return path_segments

def save_to_csv(segments, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for segment in segments:
            writer.writerow([segment[0][0].item(), segment[0][1].item(), segment[1][0].item(), segment[1][1].item()])

line_segments = generate_line_segments(400)
save_to_csv(line_segments, '200_coordinates.csv')
