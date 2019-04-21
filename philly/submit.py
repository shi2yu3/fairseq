import argparse
import urllib.request
import subprocess
import json

def submit_job(config_fn):
    cmd = "curl --ntlm --user : -X POST -H \"Content-Type: application/json\" " \
          "--data @{} https://philly/api/jobs".format(config_fn)
    return subprocess.check_output(cmd)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cluster", default="eu2", type=str, required=False,
    #                     help="Philly cluster.")
    # parser.add_argument("--vc", default="ipgsrch", type=str, required=False,
    #                     help="Philly virtual cluster.")
    # parser.add_argument("--sku", default=None, type=str, required=False,
    #                     help="How many GPUs to use.")
    # parser.add_argument("--command", default=None, type=str, required=False,
    #                     help="The command line.")

    # args, unknown_args = parser.parse_known_args()

    # config = json.load(open("philly/job.json"))

    # if args.cluster:
    #     for k in config["environmentVariables"]:
    #         config["environmentVariables"][k] = \
    #             config["environmentVariables"][k].replace(
    #                 config["metadata"]["cluster"], args.cluster)
    #     config["metadata"]["cluster"] = args.cluster
    #
    # if args.vc:
    #     for k in config["environmentVariables"]:
    #         config["environmentVariables"][k] = \
    #             config["environmentVariables"][k].replace(
    #                 config["metadata"]["vc"], args.vc)
    #     config["metadata"]["vc"] = args.vc
    #
    # if args.cluster == "wu3":
    #     config["metadata"]["queue"] = "sdrg"
    #
    # if args.sku:
    #     config["resources"]["workers"]["sku"] = args.sku
    #
    # if args.command:
    #     config["resources"]["workers"]["commandLine"] = args.command

    # json.dump(config, open("philly/job.json", "w"), indent=2)

    job_info = submit_job("philly/job.json")
    print(job_info)


if __name__ == "__main__":
    main()
