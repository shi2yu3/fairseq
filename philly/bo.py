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
args, config_file = parser.parse_known_args()
assert len(config_file) == 1
config_file = config_file[0]


def load_config(config_file):
    if os.path.isdir(config_file):
        dir = config_file
        config_file = os.path.join(dir, "exp.json")
        config = json.load(open(config_file))
    else:
        dir = os.path.join("experiment", str(uuid.uuid4()).split("-")[0])
        os.makedirs(dir, exist_ok=True)

        config = json.load(open(config_file))

        config_file = os.path.join(dir, "exp.json")
        json.dump(config, open(config_file, "w"), indent=4)

    philly_config = config["philly_config"]
    pbounds = config["pbounds"]
    return dir, philly_config, pbounds


root_dir, philly_config, pbounds = load_config(config_file)


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
        if arg in config["environmentVariables"]:
            config["environmentVariables"][arg] = kwargs[arg]
        elif arg in args_in_command:
            args_in_command[arg] = kwargs[arg]
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

    json.dump(config, open(config_file, "w"), indent=2)


def submit_job(config_fn):
    cmd = f'curl --ntlm --user : -X POST -H "Content-Type: application/json" --data @{config_fn} https://philly/api/jobs'
    job_id = subprocess.check_output(cmd)
    # job_id = b'{"jobId":"application_1553675282044_3023"}'
    print(job_id)
    job_id = job_id.decode("utf-8").replace("true", "True").replace("false", "False")
    job_id = eval(job_id)["jobId"].replace("application_", "")
    return job_id


def job_status(job_id, cluster, vc):
    cmd = f'curl -k --ntlm --user : "https://philly/api/status?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}&jobType=cust&content=full"'
    while True:
        status = subprocess.check_output(cmd)
        status = status.decode("utf-8").replace("true", "True").replace("false", "False").replace("null", "None")
        try:
            status = eval(status)
            break
        except:
            print(status)
            # raise
            time.sleep(60)
    if status["status"] == "Pass" or status["status"] == "Killed":
        return "finished"
    if status["status"] == "Running" or status["status"] == "Queued":
        return "running"
    return "failed"


def val_loss(job_id, cluster, vc):
    best_loss = 1001.0
    stdout_num = 1
    retry = 0
    while True:
        stdout = f"https://storage.{cluster}.philly.selfhost.corp.microsoft.com/{vc}/sys/jobs/application_{job_id}/stdout/{stdout_num}/stdout.txt"
        stdout = requests.get(stdout)
        if stdout.status_code == 200:
            for line in stdout.text.split("\n"):
                if 'valid on "valid" subset' in line:
                    segs = line.split(" | ")
                    epoch = str(int(segs[1].split()[1]))
                    loss = segs[3].split()[1]
                    if float(loss) < best_loss:
                        best_loss = float(loss)
                        best_epoch = epoch
                        print(f"epoch {best_epoch} best loss {best_loss}")
        else:
            retry += 1
            if retry >= 3:
                break
        stdout_num += 1
    return best_loss


def philly_job(philly_config, **kwargs):
    config_file = os.path.join(root_dir, f"{str(uuid.uuid4()).split('-')[0]}.json")
    create_config_file(config_file, philly_config, **kwargs)

    cluster = philly_config["metadata"]["cluster"]
    vc = philly_config["metadata"]["vc"]

    job_id = submit_job(config_file)

    while True:
        status = job_status(job_id, cluster, vc)
        if status == "finished":
            return -val_loss(job_id, cluster, vc)
        elif status == "running":
            time.sleep(1800)
        else:
            return None


def load_config(config_file):
    if os.path.isdir(config_file):
        config_file = os.path.join(config_file, "job.json")
        config = json.load(open(config_file))
    else:
        dir = os.path.join("experiment", str(uuid.uuid4()).split("-")[0])
        os.makedirs(dir, exist_ok=True)

        config = json.load(open(config_file))

        config_file = os.path.join(dir, "job.json")
        json.dump(config, open(config_file, "w"), indent=4)
    return config


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
    server.listen(9009)
    tornado.ioloop.IOLoop.instance().start()


def run_optimizer():
    global optimizers_config
    opt_config = optimizers_config.pop()
    philly_config = opt_config["philly_config"]
    name = opt_config["name"]
    colour_f = opt_config["colour_f"]
    colour_b = opt_config["colour_b"]

    register_data = {}
    max_target = None
    for _ in range(10):
        status = name + " wants to register: {}.\n".format(register_data)

        resp = requests.post(
            url="http://localhost:9009/bayesian_optimization",
            json=register_data,
        ).json()
        target = philly_job(philly_config, **resp)

        register_data = {
            "params": resp,
            "target": target,
        }

        if max_target is None or target > max_target:
            max_target = target

        status += name + " got {} as target.\n".format(target)
        status += name + " will to register next: {}.\n".format(register_data)
        # print(colour(status), end="\n")
        print(colored(status, colour_f, colour_b), end="\n")

    global results
    results.append((name, max_target))
    # print(colour(name + " is done!"), end="\n\n")
    print(colored(name + " is done!", colour_f, colour_b), end="\n\n")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_threads", default=1, type=int,
    #                     help="number of parallel threads")
    # args = parser.parse_args()
    # args, config_file = parser.parse_known_args()
    # assert len(config_file) == 1
    # config_file = config_file[0]

    # philly_config = Experiment(config_file)

    init()  # color
    ioloop = tornado.ioloop.IOLoop.instance()
    optimizers_config = []
    for i in range(args.num_threads):
        # optimizers_config.append({"name": f"optimizer {i}", "colour": Colours.red})  # Colours.green, Colours.blue
        optimizers_config.append({"philly_config": philly_config,
                                  "name": f"optimizer {i}",
                                  "colour_f": f_color[i % len(f_color)],
                                  "colour_b": "on_grey"})

    app_thread = threading.Thread(target=run_optimization_app)
    app_thread.daemon = True
    app_thread.start()

    targets = [run_optimizer] * args.num_threads

    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target))
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()

    results = []
    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        print(result[0], "found a maximum value of: {}".format(result[1]))

    ioloop.stop()
