import sys
import json
import shlex
import copy
import subprocess


class SearchSpace:
    def __init__(self):
        self.arguments = []

    def extend(self, arguments_to_add):
        if arguments_to_add:
            argument_to_add = arguments_to_add[0]
            if self.arguments:
                new_arguments = []
                for arg in self.arguments:
                    for val in argument_to_add['values']:
                        new_arguments.append({**arg, **{argument_to_add['name']: val}})
                self.arguments = new_arguments
            else:
                for val in argument_to_add['values']:
                    self.arguments.append({argument_to_add['name']: val})
            arguments_to_add.pop(0)
            self.extend(arguments_to_add)


def parse_command(command):
    args = shlex.split(command)
    args = parse_arguments(args)
    return args


def parse_arguments(arguments):
    args = {}
    key = ''
    for arg in arguments:
        if arg[0] == '-':
            if key:
                args[key] = ''
            key = arg
        elif key:
            args[key] = arg
            key = ''
        else:
            args[arg] = ''
    return args


def submit_job(config_fn):
    cmd = "curl --ntlm --user : -X POST -H \"Content-Type: application/json\" " \
          "--data @{} https://philly/api/jobs".format(config_fn)
    return subprocess.check_output(cmd)


def main():
    config = json.load(open(sys.argv[1]))

    philly_job_template = config['philly_config']

    args = parse_command(philly_job_template['resources']['workers']['commandLine'])

    srch_spc = SearchSpace()
    srch_spc.extend(config['args'])

    summary = []
    for arguments in srch_spc.arguments:
        philly_config = copy.deepcopy(philly_job_template)
        job_info = ['> hyper parameter']
        job_info += ['```']
        for arg in arguments:
            if arg in philly_config['environmentVariables']:
                philly_config['environmentVariables'][arg] = arguments[arg]
            elif arg in args:
                args[arg] = arguments[arg]
            else:
                raise ValueError(f'Argument {arg} is not used in command')
            job_info.append(f'{arg} {arguments[arg]}')
        commandLine = []
        for k in args:
            if isinstance(k, str) and ' ' in k:
                k = f'\'{k}\''
            commandLine.append(str(k))
            v = args[k]
            if v != '':
                if isinstance(v, str) and ' ' in v:
                    v = f'\'{v}\''
                commandLine.append(str(v))
        philly_config['resources']['workers']['commandLine'] = ' '.join(commandLine)
        job_info.append('```')

        json.dump(philly_config, open('grid_search.json', 'w'), indent=2)
        job_id = submit_job("grid_search.json")
        job_id = eval(job_id.decode('utf-8'))["jobId"].replace("application_", "")
        job_info.append(f'> job: [{job_id}](https://philly/#/job/eu2/ipgsrch/{job_id})')
        job_info.append('')
        job_info.append('| test | epoch | rouge-1 | rouge-2 | rouge-l |')
        job_info.append('| --- | --- | --- | --- | --- |')

        summary.append('\n'.join(job_info))

    summary = '\n\n'.join(summary)
    with open('grid_search.md', 'w') as f:
        f.writelines(summary)
    print(summary)


if __name__ == '__main__':
    main()
