import os
import glob


# def get_file_list(top_dir):
#     for root, dirs, files in os.walk(top_dir):
#         for dir in dirs:
#             dirPath = os.path.join(root, dir)
#             print(f'dirPath = {dirPath}')

#         for file in files:
#             filePath = os.path.join(root, file)
#             print(f'filePath = {filePath}')


def get_dir_list(top_dir):
    dirlist = []
    files = os.listdir(top_dir)
    for filename in files:
        if os.path.isdir(os.path.join(top_dir, filename)):
            dirlist.append(filename)
    return dirlist


def get_file_list(top_dir):
    return glob.glob(top_dir)
