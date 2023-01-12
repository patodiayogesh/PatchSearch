import os

print(os.getcwd())

lines = []

with open('SequenceR-Dataset/small/eval/data.prev_code', 'r') as file:
    for line in file:
        line = line.split("<s>")[1]
        lines.append(line)

write_file_to = open('../eval/a.txt', 'w')
for line in lines:
    write_file_to.write(line)