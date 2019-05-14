import sys

if __name__ == '__main__':
    file = sys.argv[1]
    len = int(sys.argv[2])

    with open(file + '.' + str(len), 'w', encoding='utf-8') as fout:
        with open(file, encoding='utf-8') as fin:
            for line in fin.readlines():
                line = ' '.join(line.split()[0:len])
                fout.writelines(line + '\n')
