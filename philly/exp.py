import sys
import json
import shlex
import copy
import subprocess
import os
import time
import uuid
import argparse

def submit_job(config_fn):
    cmd = f'curl --ntlm --user : -X POST -H "Content-Type: application/json" --data @{config_fn} https://philly/api/jobs'
    job_id = subprocess.check_output(cmd)
    # job_id = b'{"jobId":"application_1555486458178_20449"}'
    job_id = job_id.decode('utf-8').replace('true', 'True').replace('false', 'False')
    job_id = eval(job_id)["jobId"].replace("application_", "")
    return job_id


def get_job_status(job_id, cluster, vc):
    cmd = f'curl -k --ntlm --user : "https://philly/api/status?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}&jobType=cust&content=full"'
    while True:
        status = subprocess.check_output(cmd)
        status = status.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None')
        try:
            status = eval(status)
            break
        except:
            print(status)
            # raise
            time.sleep(60)
    if status['status'] == 'Pass' or status['status'] == 'Killed':
        return 'finished'
    if status['status'] == 'Running' or status['status'] == 'Queued':
        return 'running'
    return 'initailized'


class SearchSpace(object):
    def __init__(self):
        self.arguments = []

    def size(self):
        return len(self.arguments)

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


class Trial(object):
    def __init__(self, parent_dir=None, philly_config=None, args_in_command=None, args_for_tuning=None, parent_trial=None):
        if parent_dir is None or philly_config is None or args_in_command is None or args_for_tuning is None:
            return

        self.dir = os.path.join(parent_dir, str(uuid.uuid4()).split('-')[0])
        os.makedirs(self.dir, exist_ok=True)

        self.philly_metadata = philly_config['metadata']

        self.args_for_tuning = args_for_tuning
        config = copy.deepcopy(philly_config)
        for arg in args_for_tuning:
            if arg in config['environmentVariables']:
                config['environmentVariables'][arg] = args_for_tuning[arg]
            elif arg in args_in_command:
                args_in_command[arg] = args_for_tuning[arg]
            else:
                raise ValueError(f'Argument {arg} is not used in command')
        commandLine = []
        for k in args_in_command:
            if isinstance(k, str) and ' ' in k:
                k = f'\'{k}\''
            commandLine.append(str(k))
            v = args_in_command[k]
            if v != '':
                if isinstance(v, str) and ' ' in v:
                    v = f'\'{v}\''
                commandLine.append(str(v))
        config['resources']['workers']['commandLine'] = ' '.join(commandLine)

        self.parent_trial = parent_trial
        # if parent_trial and parent_trial.job_id:
        #     config = self.update_parent_job_id(config)

        self.config_file = os.path.join(self.dir, 'trial.json')
        json.dump(config, open(self.config_file, 'w'), indent=2)
        self.job_id = None
        self.status = 'initailized'

        print(f'Trial initialized: {self.config_file}\n  {self.args_of_exp()}')

    def from_status(self, trial_status):
        self.dir = trial_status['dir']
        if 'philly_metadata' in trial_status:
            self.philly_metadata = trial_status['philly_metadata']
        else:
            self.philly_metadata = {"cluster": "eu2",
                                    "name": "fairseq_tr",
                                    "username": "yushi",
                                    "vc": "ipgsrch"}
        self.args_for_tuning = trial_status['args_for_tuning']
        self.parent_trial = trial_status['parent_trial']
        self.config_file = trial_status['config_file']
        self.job_id = trial_status['job_id']
        self.status = trial_status['status']

    def run(self):
        if (not self.parent_trial or self.parent_trial.status == 'finished') and self.status == 'initailized':
            config = json.load(open(self.config_file))
            config = self.update_parent_job_id(config)
            json.dump(config, open(self.config_file, 'w'), indent=4)

            self.job_id = submit_job(self.config_file)
            if self.job_id:
                self.status = 'running'
                print(f'Trial submitted ({self.job_id}): {self.config_file}\n  {self.args_of_exp()}')

    def check(self):
        if self.job_id and self.status != 'finished' and self.status != 'gaveup':
            self.status = get_job_status(self.job_id, self.philly_metadata['cluster'], self.philly_metadata['vc'])
        if self.job_id:
            print(f'Trial status ({self.job_id}): {self.status}\n  {self.config_file}\n  {self.args_of_exp()}')

    def finished(self):
        if self.status == 'gaveup':
            return True
        self.check()
        if self.status == 'finished':
            return True
        else:
            return False

    def args_of_exp(self):
        if self.parent_trial:
            return self.parent_trial.args_of_exp() + json.dumps(self.args_for_tuning)
        else:
            return json.dumps(self.args_for_tuning)

    def update_parent_job_id(self, config):
        if self.parent_trial and self.parent_trial.job_id:
            for var in config['environmentVariables']:
                if '$PHILLY_JOB_ID' in config['environmentVariables'][var]:
                    config['environmentVariables'][var] = config['environmentVariables'][var].replace('$PHILLY_JOB_ID', self.parent_trial.job_id)
            config['resources']['workers']['commandLine'] = config['resources']['workers']['commandLine'].replace('$PHILLY_JOB_ID', self.parent_trial.job_id)
        return config


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


class Step(object):
    def __init__(self, parent_dir=None, config=None, parent_step=None):
        if parent_dir is None or config is None:
            return

        dir = parent_dir

        philly_config = config['philly_config']
        args_for_tuning = config['args_for_tuning']

        args_in_command = parse_command(philly_config['resources']['workers']['commandLine'])

        search_space = SearchSpace()
        search_space.extend(args_for_tuning)

        self.trials = []
        if parent_step:
            for parent_trial in parent_step.trials:
                for i, arguments in enumerate(search_space.arguments):
                    self.trials.append(Trial(parent_trial.dir, philly_config, args_in_command, arguments, parent_trial))
        else:
            for i, arguments in enumerate(search_space.arguments):
                self.trials.append(Trial(dir, philly_config, args_in_command, arguments))

    def from_status(self, step_status):
        self.trials = []
        for trial_status in step_status['trials']:
            trial = Trial()
            trial.from_status(trial_status)
            self.trials.append(trial)

    def run(self):
        for trial in self.trials:
            trial.run()

    def finished(self):
        finished = True
        for trial in self.trials:
            if not trial.finished():
                finished = False
        return finished


class Experiment(object):
    def __init__(self, config_file):
        self.steps = []
        self.finished_step = -1
        if os.path.isdir(config_file):
            self.resume(config_file)
        else:
            self.create(config_file)
        self.update_status()

    def create(self, config_file):
        dir = os.path.join('grid', str(uuid.uuid4()).split('-')[0])
        os.makedirs(dir, exist_ok=True)

        config = json.load(open(config_file))

        config_file = os.path.join(dir, 'exp.json')
        json.dump(config, open(config_file, 'w'), indent=4)

        self.steps = []
        for i in range(len(config['experiment_steps'])):
            step = Step(dir, config['experiment_steps'][i], None if i == 0 else self.steps[-1])
            self.steps.append(step)
            print(f'Step {i} initialized')

        self.finished_step = 0
        self.status_file = os.path.normpath(os.path.join(dir, 'status.json'))

    def resume(self, dir):
        config_file = os.path.join(dir, 'exp.json')
        config = json.load(open(config_file))

        self.status_file = os.path.normpath(os.path.join(dir, 'status.json'))
        status = json.load(open(self.status_file))

        self.finished_step = status['finished_step']
        assert os.path.normpath(self.status_file) == os.path.normpath(status['status_file'])

        self.from_status(status)

        self.check()

    def from_status(self, exp_status):
        # with open('examples/summarization/RESULT.md', encoding='utf-8') as f:
        #     args_id_map = {}
        #     args = {}
        #     valid = False
        #     for line in f.readlines():
        #         line = line.strip()
        #         if valid:
        #             if line[:2] == '--':
        #                 arg, val = line.split()
        #                 args[arg] = val
        #             if line[:6] == '> job:':
        #                 id = line.split()[2][1:20]
        #                 args_id_map[json.dumps(args)] = id
        #                 args = {}
        #         if line == '## Grid search':
        #             valid = True

        self.steps = []
        for i, step_status in enumerate(exp_status['steps']):
            step = Step()
            step.from_status(step_status)
            self.steps.append(step)
        self.link_trials()

        print(f'Experiment resumed')
        for step in self.steps:
            for trial in step.trials:
                print(f'Trial: {trial.config_file}\n  {trial.args_of_exp()}')

        # for step in self.steps:
        #     for trial in step.trials:
        #         for args in args_id_map:
        #             job_id = args_id_map[args]
        #             args = json.loads(args)
        #             shared_items = {k: trial.args_for_tuning[k] for k in trial.args_for_tuning if k in args and str(trial.args_for_tuning[k]) == args[k]}
        #             if len(shared_items) == len(trial.args_for_tuning) == len(args):
        #                 trial.job_id = job_id
        #                 break
        #         if not trial.parent_trial and not trial.job_id:
        #             print('debugging')


    def link_trials(self):
        dir_trial_map = {}
        for step in self.steps:
            for trial in step.trials:
                dir_trial_map[trial.dir] = trial
        for step in self.steps:
            for trial in step.trials:
                if trial.parent_trial:
                    trial.parent_trial = dir_trial_map[trial.parent_trial['dir']]

    def run(self):
        for i, step in enumerate(self.steps):
            if i >= self.finished_step:
                step.run()
        self.update_status()

    def check(self):
        while self.finished_step < len(self.steps) and self.steps[self.finished_step].finished():
            self.finished_step += 1

    def finished(self):
        self.check()
        if self.finished_step == len(self.steps):
            return True
        else:
            self.run()
            return False

    def update_status(self):
        with open(self.status_file, 'w') as f:
            f.write(self.to_json())

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_only', action='store_true', help='whether to only init config files')
    args, config_file = parser.parse_known_args()
    config_file = config_file[0]
    # config_file = sys.argv[1]

    exp = Experiment(config_file)
    if args.init_only:
        return

    exp.run()
    while not exp.finished():
        time.sleep(1800)


if __name__ == '__main__':
    main()
