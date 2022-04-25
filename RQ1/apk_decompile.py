import os


def get_file_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.apk'):
                    path_list.append(file_absolute_path)
    return path_list


def apk_decompile(path_apk, path_save_directory):
    """
    return path of apk decompile

    :param path_apk: <string>, path of apk.
    :param path_save_directory: <string>, path of saved directory.
    :return: <string>, path_save_file
    """
    apktool_command = 'java -jar apktool_2.5.0.jar d -s -f {path_apk} -o {path_save_file}'
    name_apk = path_apk.split('/')[-1].split('.')[0]
    name_apk_compile = name_apk + '_compile'
    path_save_file = path_save_directory + '/' + name_apk_compile
    apktool_command = apktool_command.format(path_apk=path_apk, path_save_file=path_save_file)
    os.system(apktool_command)
    return path_save_file

