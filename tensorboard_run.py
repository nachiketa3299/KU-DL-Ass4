# 한꺼번에 텐서보드를 불러오기 위한 파일
import os

def makeDir(_filename):
    return os.path.join(os.path.curdir, 'runs', _filename)

filenames = os.listdir(os.path.join(os.path.curdir, 'runs'))

# 사용하려는 명령어는 아래와 같은 꼴
# tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
command_names = []
for filename in filenames:
    name = f'{filename}'
    ddir = f'{makeDir(filename)}'
    name += ':' + ddir
    command_names.append(name)

command = f'tensorboard --logdir='
command += ','.join(command_names)
command += ' --host localhost'

os.system(f'{command}')


