#!/usr/bin/env python3

import time
import random
import argparse
import os
import uuid
import shlex
import copy
import subprocess
import glob

try:
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction
except ImportError:
    os.system('pip install --user bayesian-optimization')
    from bayes_opt import BayesianOptimization
    from bayes_opt.util import UtilityFunction

import asyncio
import threading

import json
import requests
try:
    import tornado.ioloop
    import tornado.httpserver
    from tornado.web import RequestHandler
except ImportError:
    os.system('pip install --user tornado')
    import tornado.httpserver
    from tornado.web import RequestHandler


parser = argparse.ArgumentParser()
parser.add_argument("--num_new_jobs", default=0, type=int,
                    help="number of parallel threads")
parser.add_argument("--num_rounds", default=1, type=int,
                    help="number of round")
parser.add_argument("--port", default=9009, type=int,
                    help="the localhost port number")
parser.add_argument("--new_philly_jobs", default="", type=str,
                    help="Ids of new Philly jobs that have not been tracked")
args, input_config_file = parser.parse_known_args()
assert len(input_config_file) == 1
input_config_file = input_config_file[0]


def load_config(config_file):
    if os.path.isdir(config_file):
        dir = config_file
        config_file = os.path.join(dir, "exp.json")
        config = json.load(open(config_file))
    else:
        dir = os.path.join("bayesian", str(uuid.uuid4()).split("-")[0])
        os.makedirs(dir, exist_ok=True)

        config = json.load(open(config_file))

        config_file = os.path.join(dir, "exp.json")
        json.dump(config, open(config_file, "w"), indent=4)

    philly_config = config["philly_config"]
    pbounds = config["pbounds"]
    return dir, philly_config, pbounds


root_dir, philly_config, pbounds = load_config(input_config_file)
if os.path.basename(root_dir) is "":
    root_dir = os.path.dirname(root_dir)


user, pswd = json.load(open('.auth')).values()


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.
    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    time.sleep(random.randint(1, 7))
    return -x ** 2 - (y - 1) ** 2 + 1


def parse_command(command):
    args = shlex.split(command)
    args = parse_arguments(args)
    return args


def parse_arguments(arguments):
    args = {}
    key = ""
    for arg in arguments:
        if arg[0] == "-":
            if key:
                args[key] = ""
            key = arg
        elif key:
            args[key] = arg
            key = ""
        else:
            args[arg] = ""
    return args


def fix_int_params(params):
    for p, v in params.items():
        if p in ["--warmup-updates"] and not isinstance(v, int):
            params[p] = int(v + 1 - 1e-10)
            # print(f"{p} changed from {v} to {params[p]}")
    return params


def create_config_file(config_file, philly_config, **kwargs):
    trial_id = os.path.basename(config_file).replace("_job.json", "")
    config = copy.deepcopy(philly_config)
    config["metadata"]["name"] += f"_{os.path.basename(root_dir)}_{trial_id}"
    args_in_command = parse_command(config["resources"]["workers"]["commandLine"])

    kwargs = fix_int_params(kwargs)

    for arg, val in kwargs.items():
        if arg in config["environmentVariables"]:
            config["environmentVariables"][arg] = val
        elif arg in args_in_command:
            args_in_command[arg] = val
        else:
            raise ValueError(f"Argument {arg} is not used in command")
    commandLine = []
    for k in args_in_command:
        if isinstance(k, str) and " " in k:
            k = f"'{k}'"
        commandLine.append(str(k))
        v = args_in_command[k]
        if v != "":
            if isinstance(v, str) and " " in v:
                v = f"'{v}'"
            commandLine.append(str(v))
    config["resources"]["workers"]["commandLine"] = " ".join(commandLine)

    json.dump(config, open(config_file, "w"), indent=4)


def submit_job(config_fn):
    cmd = f'curl -k --silent --ntlm --user "{user}":"{pswd}" -X POST -H "Content-Type: application/json" --data @{config_fn} https://philly/api/jobs'
    job_id = subprocess.check_output(cmd, shell=True)
    # job_id = b'{"jobId":"application_1553675282044_3023"}'
    print(job_id)
    job_id = job_id.decode("utf-8").replace("true", "True").replace("false", "False").replace("null", "None")
    job_id = eval(job_id)
    if "jobId" in job_id:
        job_id = job_id["jobId"].replace("application_", "")
        print(f"new job was submitted {job_id}\n")
    else:
        print(f"failed to submit job\n")
    return job_id


def kill_job(job_id, cluster):
    cmd = f'curl -k --silent --ntlm --user "{user}":"{pswd}" "https://philly/api/abort?clusterId={cluster}&jobId=application_{job_id}"'
    killed = subprocess.check_output(cmd, shell=True)
    # killed = b'{"phillyversion": 116, "jobkilled": "application_1553675282044_3310"}'
    print(killed)
    killed = killed.decode("utf-8").replace("true", "True").replace("false", "False").replace("null", "None")
    killed = eval(killed)
    if "jobkilled" in killed:
        print(f"job {job_id} was killed")
        return True
    else:
        print(f"failed to kill job {job_id}")
        return False


def job_status(job_id, cluster, vc):
    cmd = f'curl -k --silent --ntlm --user "{user}":"{pswd}" "https://philly/api/status?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}&jobType=cust&content=full"'
    while True:
        status = subprocess.check_output(cmd, shell=True)
        status = status.decode("utf-8").replace("true", "True").replace("false", "False").replace("null", "None")
        try:
            status = eval(status)
            break
        except:
            print(status)
            time.sleep(60)
    if status["status"] == "Pass" or status["status"] == "Killed":
        return "finished"
    if status["status"] == "Running" or status["status"] == "Queued":
        return "running"
    return "failed"


def job_metadata(job_id, cluster, vc):
    cmd = f'curl -k --silent --ntlm --user "{user}":"{pswd}" "https://philly/api/metadata?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}"'
    while True:
        metadata = subprocess.check_output(cmd, shell=True)
        metadata = metadata.decode("utf-8").replace("true", "True").replace("false", "False").replace("null", "None")
        try:
            metadata = eval(metadata)
            break
        except:
            print(metadata)
            time.sleep(60)
    return metadata


def val_loss(job_id, cluster, vc):
    best_loss = None
    stdout_num = 1
    retry = 0
    diverged_steps = 0
    while True:
        stdout = f"https://storage.{cluster}.philly.selfhost.corp.microsoft.com/{vc}/sys/jobs/application_{job_id}/stdout/{stdout_num}/stdout.txt"
        stdout = requests.get(stdout)
        if stdout.status_code == 200:
            diverged_steps = 0
            for line in stdout.text.split("\n"):
                if "valid on 'valid' subset" in line:
                    segs = line.split(" | ")
                    epoch = str(int(segs[1].split()[1]))
                    loss = segs[3].split()[1]
                    print(f"job {job_id} epoch {epoch} loss {loss}")
                    if best_loss is None or float(loss) < best_loss:
                        best_loss = float(loss)
                        best_epoch = epoch
                        diverged_steps = 0
                    else:
                        diverged_steps += 1
                        if diverged_steps >= 4:
                            break
            if best_loss is not None:
                print("")
            if diverged_steps >= 4:
                break
        else:
            retry += 1
            if retry >= 3:
                break
        stdout_num += 1
    return best_loss, diverged_steps >= 4


def update_job(job_id, cluster, vc, trial_id, wait, **kwargs):
    info_file = os.path.join(root_dir, f"{trial_id}_info.json")

    job_info = {"params": kwargs,
                "trial_id": trial_id,
                "philly_id": job_id,
                "cluster": cluster,
                "vc": vc,
                "status": "running",
                "target": None}

    while True:
        status = job_status(job_id, cluster, vc)
        if status == "finished":
            loss, diverged = val_loss(job_id, cluster, vc)
            if loss is not None:
                job_info["target"] = -loss
                job_info["status"] = "finished"
            else:
                job_info["status"] = "failed"
            json.dump(job_info, open(info_file, "w"), indent=4)
            break
        elif status == "running":
            loss, diverged = val_loss(job_id, cluster, vc)
            if loss is not None:
                job_info["target"] = -loss
            if diverged and kill_job(job_id, cluster):
                    job_info["status"] = "finished" if loss is not None else "failed"
            json.dump(job_info, open(info_file, "w"), indent=4)
            if not diverged and wait and args.num_rounds > 1:
                time.sleep(random.randint(900, 18000))
            else:
                break
        else:
            job_info["status"] = "failed"
            json.dump(job_info, open(info_file, "w"), indent=4)
            break
        print(f"job {job_id} status: {job_info['status']}")

    return job_info


def philly_job(philly_config, **kwargs):
    trial_id = str(uuid.uuid4()).split('-')[0]

    config_file = os.path.join(root_dir, f"{trial_id}_job.json")
    create_config_file(config_file, philly_config, **kwargs)

    cluster = philly_config["metadata"]["cluster"]
    vc = philly_config["metadata"]["vc"]

    job_id = submit_job(config_file)

    return update_job(job_id, cluster, vc, trial_id, True, **kwargs)


class BayesianOptimizationHandler(RequestHandler):
    """Basic functionality for NLP handlers."""
    _bo = BayesianOptimization(
        f=philly_job,
        # pbounds={"x": (-4, 4), "y": (-3, 3)}
        pbounds=pbounds
    )
    _uf = UtilityFunction(kind="ucb", kappa=3, xi=1)

    def post(self):
        """Deal with incoming requests."""
        body = tornado.escape.json_decode(self.request.body)
        suggest = body.pop("suggest")

        try:
            self._bo.register(
                params=body["params"],
                target=body["target"],
            )
            status = f"BO registered: {body}.\n"
            status += f"BO has registered: {len(self._bo.space)} points.\n"
            print(status)
        except KeyError:
            pass
        finally:
            suggested_params = {}
            if suggest:
                suggested_params = self._bo.suggest(self._uf)

        self.write(json.dumps(suggested_params))


def run_optimization_app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    handlers = [
        (r"/bayesian_optimization", BayesianOptimizationHandler),
    ]
    server = tornado.httpserver.HTTPServer(
        tornado.web.Application(handlers)
    )
    server.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()


def swap_params(params):
    if "--max-lr" in params and "--lr" in params:
        if params["--max-lr"] < params["--lr"]:
            params["--max-lr"], params["--lr"] = params["--lr"], params["--max-lr"]
            # print(f"--max-lr and --lr are swapped")
    return params


def run_optimizer():
    global optimizers_config
    opt_config = optimizers_config.pop()
    name = opt_config["name"]

    register_data = {"suggest": True}
    max_target = None
    if "philly_config" in opt_config:
        while len(jobs_to_resume) > 0:
            time.sleep(random.randint(1, 10))
        time.sleep(random.randint(60, 120))

        philly_config = opt_config["philly_config"]

        resp = requests.post(
            url=f"http://localhost:{args.port}/bayesian_optimization",
            json=register_data,
        ).json()

        for _ in range(args.num_rounds):
            resp = swap_params(resp)
            print(f"{name} is trying {resp}")

            job_info = philly_job(philly_config, **resp)
            # job_info = {"status": "running", "target": None}

            target = None if job_info["status"] == "running" else job_info["target"]
            register_data = {} if target is None else {"params": resp,
                                                       "target": target,
                                                       "suggest": True}

            if target is not None:
                if max_target is None or target > max_target:
                    max_target = target

            if len(register_data) > 0:
                resp = requests.post(
                    url=f"http://localhost:{args.port}/bayesian_optimization",
                    json=register_data,
                ).json()
                print(f"{name} registered: {register_data}.\n")
    elif "job_info" in opt_config:
        job_info = opt_config["job_info"]
        target = job_info["target"]

        jobs_to_resume.remove(name)
        print(f"({len(jobs_to_resume)} jobs left to resume)")

        if job_info["status"] != "running" and target is not None:
            register_data = {
                "params": job_info["params"],
                "target": target,
                "suggest": False}
        elif job_info["status"] != "failed":
            job_info = update_job(job_info["philly_id"],
                                  job_info["cluster"],
                                  job_info["vc"],
                                  job_info["trial_id"],
                                  False if args.num_new_jobs == 0 or args.num_rounds <= 1 else True,
                                  **job_info["params"])

            target = None if job_info["status"] == "running" else job_info["target"]
            register_data = {} if target is None else {"params": job_info["params"],
                                                       "target": target,
                                                       "suggest": False}

        if target is not None:
            if max_target is None or target > max_target:
                max_target = target

        if args.num_new_jobs > 0 and len(register_data) > 0:
            requests.post(
                url=f"http://localhost:{args.port}/bayesian_optimization",
                json=register_data,
            )
            print(f"{name} registered: {register_data}.")
    else:
        raise ValueError("Optimizers config should contain 'philly_config' or 'job_info'")

    results.append((name, max_target))
    status = f"{name} is done!"
    # if "job_info" in opt_config:
    #     jobs_to_resume.remove(name)
    #     status += f" ({len(jobs_to_resume)} jobs left to register)"
    status += "\n"
    print(status)


def track_new_jobs(new_jobs):
    if new_jobs is not []:
        for job_id in new_jobs:
            job_info = job_metadata(job_id, philly_config['metadata']['cluster'], philly_config['metadata']['vc'])
            exp_id, trial_id = job_info['name'].split('!')[0].split('_')[-2:]
            assert os.path.basename(root_dir) == exp_id

            args_in_command = parse_command(job_info['cmd'])
            param = {}
            for p in pbounds:
                param[p] = args_in_command[p]

            config_file = os.path.join(root_dir, f"{trial_id}_job.json")
            if not os.path.exists(config_file):
                create_config_file(config_file, philly_config, **param)
            update_job(job_id, philly_config['metadata']['cluster'], philly_config['metadata']['vc'], trial_id, False, **param)

def resume_job_info(exp_dir):
    existing_jobs = []
    if os.path.isdir(exp_dir):
        info_files = glob.glob(f"{exp_dir}/*_info.json")
        for info_file in info_files:
            print(f"resuming {info_file}")
            info = json.load(open(info_file))
            if info["status"] == "failed":
                job_file = info_file.replace('_info.json', '_job.json')
                if os.path.exists(job_file):
                    os.rename(job_file, f"{job_file}.bad")
                os.rename(info_file, f"{info_file}.bad")
            else:
                for p in info["params"]:
                    info["params"][p] = float(info["params"][p])
                existing_jobs.append(info)
    return existing_jobs

results = []
jobs_to_resume = []

if __name__ == "__main__":
    ioloop = tornado.ioloop.IOLoop.instance()

    optimizers_config = []

    track_new_jobs(args.new_philly_jobs.split())
    existing_jobs = resume_job_info(input_config_file)
    # existing_jobs = existing_jobs[0:2]
    for i, job_info in enumerate(existing_jobs):
        # if job_info["trial_id"] != "21b49dc6":
        #     continue
        if job_info["status"] != "failed":
            optimizers_config.append({"job_info": job_info,
                                      "name": f"optimizer {job_info['trial_id']}"})
            jobs_to_resume.append(f"optimizer {job_info['trial_id']}")
    print(f"\nResuming {len(jobs_to_resume)} targets for registering\n")

    for i in range(args.num_new_jobs):
        optimizers_config.append({"philly_config": philly_config,
                                  "name": f"optimizer {i}"})

    app_thread = threading.Thread(target=run_optimization_app)
    app_thread.daemon = True
    app_thread.start()

    targets = [run_optimizer] * len(optimizers_config)

    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target))
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()


    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    max_target = -1e10
    for result in results:
        print(f"{result[0]} found a maximum value of: {result[1]}")
        if result[1] is not None and max_target < result[1]:
            max_target = result[1]
    print(f"\nMaximum value: {max_target}")

    ioloop.stop()
