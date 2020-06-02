import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

image_dir = current_dir + "/Images/output/"
label_dir = current_dir + "/Labels/output/"

# Create and/or truncate train.txt and test.txt
file_train = open('train2.txt', 'w')

# Populate train.txt and test.txt
for pathAndFilename in glob.iglob(os.path.join(label_dir, "*.txt")):
    f = open(pathAndFilename, 'r')
    lines = f.readlines()
    f.close()
    line_length = list(range(len(lines)))
    for line in lines:
        line = line.replace(' ',',')
        temp = line.split(',')
        if temp[5] != "0\n":
            file_train.write(line)
file_train.close()
