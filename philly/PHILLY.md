# upload data
```
powershell philly\philly-fs.ps1 -cp -r data-bin/cnndm //philly/eu2/ipgsrch/yushi/fairseq/data-bin
```

# submit job
```
curl -k --ntlm --user : -X POST -H "Content-Type: application/json" --data @job.json https://philly/api/jobs
```

# monitor job
```
curl -k --ntlm --user : "https://philly/api/status?clusterId=eu2&vcId=ipgsrch&jobId=application_1555486458178_13653&jobType=cust&content=full"
```

# abort job
```
curl -k --ntlm --user : "https://philly/api/abort?clusterId=eu2&jobId=application_1450168330223_0011"
```

# list jobs
```
curl -k --ntlm --user : "https://philly/api/list?clusterId=eu2&vcId=ipgsrch"
```

# get job metadata
```
curl -k --ntlm --user : "https://philly/api/metadata?clusterId=eu2&vcId=ipgsrch&jobId=application_1555486458178_13653"
```

# To run bo.py in background on linux machine

Requirement: Python 3.6

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9000 bayesian/07e509b8/ > 07e509b8.log &

nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9001 bayesian/11853456/ > 11853456.log &
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9005 bayesian/485a920e/ > 485a920e.log &

nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9002 bayesian/9394bf00/ > 9394bf00.log &
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9004 bayesian/ab020539/ > ab020539.log &

nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9003 bayesian/e3f5b02d/ > e3f5b02d.log &
```

Find nohup process

```
ps ax | grep bo.py
```

Kill the process

```
kill PID
or
pkill -f bo.py
```