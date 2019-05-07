import time
import random
from colorama import init
from termcolor import colored
import argparse
import os
import uuid
import shlex
import copy
import subprocess
import glob

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction, Colours

import asyncio
import threading

try:
    import json
    import tornado.ioloop
    import tornado.httpserver
    from tornado.web import RequestHandler
    import requests
except ImportError:
    raise ImportError(
        "In order to run this example you must have the libraries: " +
        "`tornado` and `requests` installed."
    )

# f_color = ["grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
f_color = ["red", "green", "yellow", "blue", "magenta", "cyan"]
b_color = ["on_grey", "on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan", "on_white"]


parser = argparse.ArgumentParser()
parser.add_argument("--num_threads", default=1, type=int,
                    help="number of parallel threads")
parser.add_argument("--localhost", default=9009, type=int,
                    help="the local host")
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


def create_config_file(config_file, philly_config, **kwargs):
    config = copy.deepcopy(philly_config)
    args_in_command = parse_command(config["resources"]["workers"]["commandLine"])

    for arg in kwargs:
        val = kwargs[arg]
        if arg == "--warmup-updates":
            val = int(val)

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
    cmd = f'curl --silent --ntlm --user : -X POST -H "Content-Type: application/json" --data @{config_fn} https://philly/api/jobs'
    job_id = subprocess.check_output(cmd)
    # job_id = b'{"jobId":"application_1553675282044_3023"}'
    print(f"new job submitted {job_id}")
    job_id = job_id.decode("utf-8").replace("true", "True").replace("false", "False")
    job_id = eval(job_id)["jobId"].replace("application_", "")
    return job_id


def job_status(job_id, cluster, vc):
    cmd = f'curl -k --silent --ntlm --user : "https://philly/api/status?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}&jobType=cust&content=full"'
    while True:
        status = subprocess.check_output(cmd)
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


def val_loss(job_id, cluster, vc):
    best_loss = None
    stdout_num = 1
    retry = 0
    while True:
        stdout = f"https://storage.{cluster}.philly.selfhost.corp.microsoft.com/{vc}/sys/jobs/application_{job_id}/stdout/{stdout_num}/stdout.txt"
        stdout = requests.get(stdout)
        if stdout.status_code == 200:
            for line in stdout.text.split("\n"):
                if "valid on 'valid' subset" in line:
                    segs = line.split(" | ")
                    epoch = str(int(segs[1].split()[1]))
                    loss = segs[3].split()[1]
                    if best_loss is None or float(loss) < best_loss:
                        best_loss = float(loss)
                        best_epoch = epoch
                        print(f"epoch {best_epoch} best loss {best_loss}")
        else:
            retry += 1
            if retry >= 3:
                break
        stdout_num += 1
    return best_loss


def wait_for(job_id, cluster, vc, trial_id, **kwargs):
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
            loss = val_loss(job_id, cluster, vc)
            if loss is not None:
                job_info["target"] = -loss
            job_info["status"] = "finished"
            json.dump(job_info, open(info_file, "w"), indent=4)
            break
        elif status == "running":
            print(f"job {job_id} is still running")
            json.dump(job_info, open(info_file, "w"), indent=4)
            time.sleep(1800)
        else:
            job_info["status"] = "failed"
            json.dump(job_info, open(info_file, "w"), indent=4)
            break

    return job_info["target"]


def philly_job(philly_config, **kwargs):
    trial_id = str(uuid.uuid4()).split('-')[0]

    config_file = os.path.join(root_dir, f"{trial_id}_job.json")
    create_config_file(config_file, philly_config, **kwargs)

    cluster = philly_config["metadata"]["cluster"]
    vc = philly_config["metadata"]["vc"]

    job_id = submit_job(config_file)

    return wait_for(job_id, cluster, vc, trial_id, **kwargs)


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

        try:
            self._bo.register(
                params=body["params"],
                target=body["target"],
            )
            print("BO has registered: {} points.".format(len(self._bo.space)), end="\n\n")
        except KeyError:
            pass
        finally:
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
    server.listen(args.localhost)
    tornado.ioloop.IOLoop.instance().start()


def run_optimizer():
    global optimizers_config
    opt_config = optimizers_config.pop()
    name = opt_config["name"]
    colour_f = opt_config["colour_f"]
    colour_b = opt_config["colour_b"]

    register_data = {}
    max_target = None
    if "philly_config" in opt_config:
        philly_config = opt_config["philly_config"]

        for _ in range(10):
            status = name + " wants to register: {}.\n".format(register_data)

            resp = requests.post(
                url=f"http://localhost:{args.localhost}/bayesian_optimization",
                json=register_data,
            ).json()
            target = philly_job(philly_config, **resp)

            if target is None:
                register_data = {}

                status += name + " haven't got target.\n"
            else:
                register_data = {
                    "params": resp,
                    "target": target,
                }

                if max_target is None or target > max_target:
                    max_target = target

                status += name + " got {} as target.\n".format(target)

            status += name + " will to register next: {}.\n".format(register_data)
            print(colored(status, colour_f, colour_b), end="\n")
    elif "job_info" in opt_config:
        job_info = opt_config["job_info"]

        status = name + " wants to register: {}.\n".format(register_data)

        requests.post(
            url=f"http://localhost:{args.localhost}/bayesian_optimization",
            json=register_data,
        )

        target = job_info["target"]
        if target is not None:
            register_data = {
                "params": job_info["params"],
                "target": target,
            }
        elif job_info["status"] != "failed":
            target = wait_for(job_info["philly_id"],
                              job_info["cluster"],
                              job_info["vc"],
                              job_info["trial_id"],
                              **job_info["params"])

            if target is None:
                register_data = {}
                status += name + " haven't got target.\n"
            else:
                register_data = {
                    "params": job_info["params"],
                    "target": target,
                }

            if max_target is None or target > max_target:
                max_target = target

            status += name + " got {} as target.\n".format(target)
        status += name + " will to register next: {}.\n".format(register_data)
        print(colored(status, colour_f, colour_b), end="\n")
    else:
        raise ValueError("Optimizers config should contain 'philly_config' or 'job_info'")


    # global results
    results.append((name, max_target))
    print(colored(name + " is done!", colour_f, colour_b), end="\n\n")


def resume_job_info(config_file):
    job_info = []
    if os.path.isdir(config_file):
        dir = config_file
        info_files = glob.glob(f"{dir}/*_info.json")
        for info_file in info_files:
            job_info.append(json.load(open(info_file)))
    return job_info

results = []

if __name__ == "__main__":
    init()  # init color
    ioloop = tornado.ioloop.IOLoop.instance()

    optimizers_config = []

    existing_jobs = resume_job_info(input_config_file)
    for i, job_info in enumerate(existing_jobs):
        if job_info["status"] != "failed":
            optimizers_config.append({"job_info": job_info,
                                      "name": f"optimizer {job_info['trial_id']}",
                                      "colour_f": f_color[(i + args.num_threads) % len(f_color)],
                                      "colour_b": "on_white"})

    for i in range(args.num_threads):
        optimizers_config.append({"philly_config": philly_config,
                                  "name": f"optimizer {i}",
                                  "colour_f": f_color[i % len(f_color)],
                                  "colour_b": "on_grey"})

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

    for result in results:
        print(result[0], "found a maximum value of: {}".format(result[1]))

    ioloop.stop()
