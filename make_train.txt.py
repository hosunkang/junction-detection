import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

image_dir = current_dir + "/Images/output/"
label_dir = current_dir + "/Labels/output/"

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')

# Populate train.txt and test.txt
for pathAndFilename in glob.iglob(os.path.join(label_dir, "*.txt")):
    f = open(pathAndFilename, 'r')
    lines = f.readlines()
    f.close()
    line_length = list(range(len(lines)))
    for line,i in zip(lines,line_length):
        line = line.split(' ')
        if i == 0:
            file_train.write(line[0] + ' ')
        line[1] = line[1].replace("\n", " ")
        file_train.write(line[1])
    file_train.write("\n")
file_train.close()
