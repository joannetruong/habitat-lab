import subprocess
import sys
import os

sbatch_file = sys.argv[1]
args = sys.argv[2:]

dont_save = False
if '-d' in args:
    files = [a for a in args if a != '-d']
    dont_save = True

with open(sbatch_file) as f:
    data = f.read()

for idx, arg in enumerate(args):
    data = data.replace(f'${idx+1}', arg)

# Create and delete a temporary file
temp_file = sbatch_file.replace('.sh', '_'+'_'.join(args)+'.sh')

with open(temp_file, 'w') as f:
    f.write(data)
try:
    subprocess.check_call(['sbatch', temp_file])
except:
    print('sbatch failed!!')

if dont_save:
    os.remove(temp_file)