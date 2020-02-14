from multiprocessing import Pool
import glob

filenames = glob.glob('/home/chalkidis/common_crawl_shards/*')
filenames = glob.glob('/home/chalkidis/wikipedia_shards/*')
filenames = glob.glob('/home/chalkidis/europarl_shards/*')

sum_filenames = len(filenames)
print('INPUT FILES: ', sum_filenames)


def process_file(filename):
    lines = []
    with open(filename) as file:
        for line in file.readlines():
            lines.append(line.encode("utf-8", "ignore").decode())

    with open(filename, 'w') as file:
        for line in lines:
            if line.endswith('\n'):
                line = line.strip('\n')
            file.write(line + '\n')


with Pool(processes=10) as pool:
    pool.map(process_file, filenames)
