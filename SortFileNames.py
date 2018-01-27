import re
import glob
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for infile in sorted(glob.glob('*.csv'), key=numericalSort):
    print ("Current File Being Processed is: " + infile)