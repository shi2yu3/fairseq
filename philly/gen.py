import argparse
import datetime
import subprocess
import json
import os
import requests

def job_exists(job_id, cluster, vc):
    print(f'\n++++ Trying to find {job_id} in cluster {cluster} vc {vc}')
    cmd = f'curl -k --ntlm --user : "https://philly/api/status?clusterId={cluster}&vcId={vc}&jobId=application_{job_id}&jobType=cust&content=full"'
    status = subprocess.check_output(cmd)
    status = status.decode('utf-8').replace('true', 'True').replace('false', 'False').replace('null', 'None')
    status = eval(status)
    if 'vc' in status and status['vc'] == vc:
        print(f'{job_id} found in cluster {cluster} vc {vc}')
        return True
    else:
        print(f'{job_id} not found in cluster {cluster} vc {vc}')
        return False


def valid_loss(job_id, epochs, cluster, vc):
    best_loss = None
    if '_best' in epochs:
        best_loss = 1000.0
        epochs = epochs.copy()
        epochs.remove('_best')
    max_epoch = max([int(e) for e in epochs]) if epochs else 0

    valid_loss = {}
    stdout_num = 1
    retry = 0
    while True:
        # print(f'stdout/{retry}/stdout.txt')
        stdout = f'https://storage.{cluster}.philly.selfhost.corp.microsoft.com/{vc}/sys/jobs/application_{job_id}/stdout/{stdout_num}/stdout.txt'
        stdout = requests.get(stdout)
        if not stdout.ok:
            retry += 1
            if retry >= 5:
                break
            else:
                stdout_num += 1
                continue
        for line in stdout.text.split('\n'):
            if "valid on 'valid' subset" in line:
                segs = line.split(' | ')
                epoch = str(int(segs[1].split()[1]))
                loss = segs[3].split()[1]
                if epoch in epochs:
                    valid_loss[epoch] = loss
                    print(f'epoch {epoch} loss {loss}')
                if best_loss is not None and float(loss) < best_loss:
                    best_loss = float(loss)
                    print(f'epoch {epoch} best loss {best_loss}')
                if best_loss is None and int(epoch) >= max_epoch:
                    break
        if best_loss or len(valid_loss) < len(epochs):
            stdout_num += 1
        else:
            break
    for epoch in epochs:
        if epoch not in valid_loss:
            valid_loss[epoch] = '---'
    return valid_loss, best_loss


def create_config(train_job_id, epoch, cluster, vc, queue):
    config = {
        'version': str(datetime.date.today()),
        'metadata': {
            'name': 'fairseq_gen',
            'cluster': cluster,
            'vc': vc,
            'username': 'yushi'
        },
        'environmentVariables': {
            'rootdir': f'/philly/{cluster}/{vc}/yushi/fairseq',
            'datadir': 'data-bin/cnndm',
            'arch': 'transformer_vaswani_wmt_en_de_big',
            'modelpath': f'/var/storage/shared/{vc}/sys/jobs/application_{train_job_id}',
            'epoch': f'{epoch}'
        },
        'resources': {
            'workers': {
                'type': 'skuResource',
                'sku': 'G1',
                'count': 1,
                'image': 'phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36-vcr',
                'commandLine': 'python $rootdir/generate.py $rootdir/$datadir --path $modelpath/checkpoint$epoch.pt --batch-size 64 --beam 5 --remove-bpe --no-repeat-ngram-size 3 --print-alignment --output_dir $PHILLY_JOB_DIRECTORY --min-len 60'
            }
        }
    }
    if queue:
        config['metadata']['queue'] = queue
    json.dump(config, open('gen.json', 'w'), indent=4)
    return 'gen.json'


def submit_job(config_fn):
    cmd = f'curl --ntlm --user : -X POST -H "Content-Type: application/json" --data @{config_fn} https://philly/api/jobs'
    job_id = subprocess.check_output(cmd)
    print(job_id)
    job_id = job_id.decode('utf-8').replace('true', 'True').replace('false', 'False')
    job_id = eval(job_id)["jobId"].replace("application_", "")
    # job_id = ‘1553675282044_3052’
    if job_id:
        print(f'job submitted as {job_id}')
    else:
        print('job submission failed')
    return job_id


def test_exp(status_file, epochs, args):
    status = json.load(open(status_file))

    trials = status['steps'][0]['trials']

    args_for_tuning = {}
    for trial in trials:
        if trial['status'] == 'gaveup':
            continue
        for arg in trial['args_for_tuning']:
            if arg in args_for_tuning:
                args_for_tuning[arg].append(trial['args_for_tuning'][arg])
            else:
                args_for_tuning[arg] = [trial['args_for_tuning'][arg]]
    concerned_args = []
    for arg in args_for_tuning:
        if len(set(args_for_tuning[arg])) > 1:
            concerned_args.append(arg)

    md_table = f'|'
    for arg in concerned_args:
        md_table += f' {arg} |'
    md_table += f' train | epoch | loss | test | r-1 | r-2 | r-3 |'
    md_table += f'\n|'
    md_table += f' --- |' * (len(concerned_args) + 7)

    for trial in trials:
        if trial['status'] == 'gaveup':
            continue
        train_job_id = trial['job_id']
        print(f'\n\n==== Training Job {train_job_id} ====')

        cluster = trial['philly_metadata']['cluster']
        vc = trial['philly_metadata']['vc']
        queue = trial['philly_metadata']['queue'] if 'queue' in trial['philly_metadata'] else None

        loss, best_loss = valid_loss(train_job_id, epochs, cluster, vc)

        for epoch in epochs:
            if args.dryrun:
                job_id = '---'
            else:
                print(f'\n++++ Epoch {epoch}')
                if epoch == '_best' and best_loss == 1000.0:
                    print(f'no best epoch')
                else:
                    config_file = create_config(train_job_id, epoch, cluster, vc, queue)
                    job_id = submit_job(config_file)

            md_table += f'\n|'
            for arg in concerned_args:
                md_table += f' {str(trial["args_for_tuning"][arg])} |'
            md_table += f' [{train_job_id}](https://philly/#/job/{cluster}/{vc}/{train_job_id}) |'
            md_table += f' {epoch} |'
            if epoch == '_best':
                md_table += f' {best_loss} |'
            else:
                md_table += f' {loss[epoch]} |'
            md_table += f' [{job_id}](https://philly/#/job/{cluster}/{vc}/{job_id}) |'
            md_table += f' --- |'
            md_table += f' --- |'
            md_table += f' --- |'

    print(f'\n\n==== Markdown Table ====\n\n{md_table}')


def test_jobs(train_jobs, epochs, args):
    resource = [{"cluster": "eu2", "vc": "ipgsrch", "queue": None},
                {"cluster": "wu3", "vc": "ipgsp", "queue": "sdrg"},
                {"cluster": "wu3", "vc": "msrmt", "queue": "sdrg"},
                {"cluster": "rr1", "vc": "ipgsrch", "queue": None}]

    md_table = '| train | epoch | loss | test | r-1 | r-2 | r-3 |'
    md_table += '\n| --- | --- | --- | --- | --- | --- | --- |'

    for train_job_id in train_jobs:
        print(f'\n\n==== Training Job {train_job_id} ====')
        cluster = None
        vc = None
        queue = None
        for i in range(len(resource)):
            cluster = resource[i]["cluster"]
            vc = resource[i]["vc"]
            queue = resource[i]["queue"]

            if job_exists(train_job_id, cluster, vc):
                break

        loss, best_loss = valid_loss(train_job_id, epochs, cluster, vc)

        for epoch in epochs:
            if args.dryrun:
                job_id = '---'
            else:
                print(f'\n++++ Epoch {epoch}')
                config_file = create_config(train_job_id, epoch, cluster, vc, queue)
                job_id = submit_job(config_file)

            md_table += f'\n|'
            md_table += f' [{train_job_id}](https://philly/#/job/{cluster}/{vc}/{train_job_id}) |'
            md_table += f' {epoch} |'
            if epoch == '_best':
                md_table += f' {best_loss} |'
            else:
                md_table += f' {loss[epoch]} |'
            md_table += f' [{job_id}](https://philly/#/job/{cluster}/{vc}/{job_id}) |'
            md_table += f' --- |'
            md_table += f' --- |'
            md_table += f' --- |'

    print(f'\n\n==== Markdown Table ====\n\n{md_table}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default='_best', type=str,
                        help='epoch to test')
    parser.add_argument('--dryrun', action='store_true',
                        help='do everything except submit philly jobs')
    # parser.add_argument(dest='train_jobs',
    #                     help='train job ids or exp status file')
    args, train_jobs = parser.parse_known_args()

    epochs = args.epoch.split()

    if len(train_jobs) == 1 and os.path.isfile(train_jobs[0]):
        test_exp(train_jobs[0], epochs, args)
    else:
        test_jobs(train_jobs, epochs, args)


if __name__ == '__main__':
    main()
