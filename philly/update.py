import argparse
import subprocess
import os

FILE_MAP = {"train.py": ""}
FOLDER_MAP = {"data-bin/cnndm": "data-bin",
              "fairseq": ""}

def mkdir(dir):
    cmd = "powershell.exe {} -mkdir {}".format(philly_tool, dir)
    fail = subprocess.call(cmd)
    print("[{}] philly-fs -mkdir {}".format("failed" if fail else "succeeded", dir))

    cmd = "powershell.exe {} -chmod 777 {}".format(philly_tool, dir)
    fail = subprocess.call(cmd)
    print("[{}] philly-fs -chmod 777 {}".format("failed" if fail else "succeeded", dir))


def copy_file(file, target_folder):
    cmd = "powershell.exe {} -cp {} {}".format(philly_tool, file, target_folder)
    fail = subprocess.call(cmd)
    print("[{}] philly-fs -cp {} {}".format("failed" if fail else "succeeded", file, target_folder))


def copy_folder(folder, target_folder):
    cmd = "powershell.exe {} -cp -r {} {}".format(philly_tool, folder, target_folder)
    fail = subprocess.call(cmd)
    print("[{}] philly-fs -cp -r {} {}".format("failed" if fail else "succeeded", folder, target_folder))


def delete_file(file):
    cmd = "powershell.exe {} -rm -r {}".format(philly_tool, file)
    fail = subprocess.call(cmd)
    print("[{}] philly-fs -rm -r {}".format("failed" if fail else "succeeded", file))


def update_one_vc(cluster, vc, files, delete):
    philly_dir = "//philly/{}/{}/yushi/{}".format(cluster, vc, philly_folder)
    if delete:
        for file in files:
            delete_file("{}/{}".format(philly_dir, file))
    else:
        if files:
            subdirs = []
            for file in files:
                if not file in FILE_MAP and \
                    not file in FOLDER_MAP and \
                        not os.path.split(file)[0] in FOLDER_MAP:
                    print("Warning: {} is not in original list".format(file))
                subdirs.append(os.path.dirname(file))
            subdirs = set(subdirs)
        else:
            subdirs = set(list(FILE_MAP.values()) + list(FOLDER_MAP.values()))
        for subdir in subdirs:
            mkdir("{}/{}".format(philly_dir, subdir))

        if files:
            for file in files:
                if os.path.isdir(file):
                    copy_folder(file, "{}/{}".format(philly_dir, os.path.dirname(file)))
                else:
                    copy_file(file, "{}/{}".format(philly_dir, os.path.dirname(file)))
        else:
            for file in FILE_MAP.keys():
                copy_file(file, "{}/{}".format(philly_dir, FILE_MAP[file]))
            for folder in FOLDER_MAP.keys():
                copy_folder(folder, "{}/{}".format(philly_dir, FOLDER_MAP[folder]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--philly_tool", default="./philly/philly-fs.ps1", type=str,
                        required=False, help="The powershell tool of Philly.")
    parser.add_argument("--cluster", default="eu2", type=str, required=False,
                        help="Philly cluster.")
    parser.add_argument("--vc", default="ipgsrch", type=str, required=False,
                        help="Philly virtual cluster.")
    parser.add_argument("--philly_folder", default="fairseq",
                        type=str, required=False,
                        help="The storage folder on Philly.")
    parser.add_argument("--delete", default=False, action='store_true',
                        help="Whether to delete the file.")
    parser.add_argument("--file", default=["train.py"], type=str, nargs='+',
                        required=False, help="Files to upload.")
    parser.add_argument("--all", default=False, action='store_true',
                        help="Whether to update all defined by FILE_MAP and FOLDER_MAP.")
    args = parser.parse_args()

    global philly_folder, philly_tool
    philly_folder = args.philly_folder
    philly_tool = args.philly_tool.replace("\\", "/")

    for i in range(len(args.file)):
        args.file[i] = args.file[i].replace("\\", "/")

    if args.all:
        for f in {**FILE_MAP, **FOLDER_MAP}:
            if f not in args.file:
                args.file.append(f)

    update_one_vc(args.cluster, args.vc, args.file, args.delete)

if __name__ == "__main__" and __package__ is None:
    __package__ = "..pytorch_pretrained_bert"
    main()
