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
# fixed lr scheduler
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9001 bayesian/11853456/ > 11853456.log &
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9005 bayesian/485a920e/ > 485a920e.log &

# cosine lr scheduler
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9000 bayesian/07e509b8/ > 07e509b8.log &

# inverse_sqrt lr scheduler
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9002 bayesian/9394bf00/ > 9394bf00.log &
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9004 bayesian/ab020539/ > ab020539.log &

# triangular lr scheduler
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

Generation

```
# for exp 11853456, 485a920e, 07e509b8, 9394bf00, ab020539, e3f5b02d, respecitvely
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_3404 1553675282044_8686 1553675282044_4232 1553675282044_3784 1553675282044_5446 1553675282044_4778
```

Scoring

```
# exp 11853456
bash philly/score.sh 1553675282044_3986 1553675282044_3987 1553675282044_3499 1553675282044_3538 1553675282044_3502 1553675282044_3539 1553675282044_3490 1553675282044_3540 1553675282044_3503

# exp 485a920e
bash philly/score.sh 1553675282044_10211 1553675282044_10212 1553675282044_10213 1553675282044_10214 1553675282044_10215 1553675282044_10216 1553675282044_10217 1553675282044_10218 1553675282044_10219

# exp 07e509b8
bash philly/score.sh 1553675282044_10220 1553675282044_10221 1553675282044_10222 1553675282044_10223 1553675282044_10224 1553675282044_10225 1553675282044_10226 1553675282044_10227 1553675282044_10228

# exp 9394bf00
bash philly/score.sh 1553675282044_3982 1553675282044_3983 1553675282044_3984 1553675282044_3985 1553675282044_3964 1553675282044_3965 1553675282044_3966 1553675282044_3967

# exp ab020539
bash philly/score.sh 1553675282044_10229 1553675282044_10230 1553675282044_10231 1553675282044_10232 1553675282044_10233 1553675282044_10234 1553675282044_10235 1553675282044_10236 1553675282044_10237

# exp e3f5b02d
bash philly/score.sh 1553675282044_10238 1553675282044_10239 1553675282044_10240 1553675282044_10241 1553675282044_10242 1553675282044_10243 1553675282044_10244 1553675282044_10245 1553675282044_10246
```
