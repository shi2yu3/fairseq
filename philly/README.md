# Bayesian search on top of Philly

The following functions may need to change for different tasks:
- fix_int_params()
- val_loss()
- swap_params()

**First run**
```
python bo.py --num_new_jobs 6 --num_rounds 10 bo.json
```

It will generate a folder with job json and job info, say bayesian/07e509b8, and submit 6 jobs to Philly.

**To track and update the jobs**
```
python bo.py bayesian/07e509b8
```

If there are some Philly jobs that were submitted last time but have not been tracked, use `--new_philly_jobs` followed by a list of Philly ids, for example:
```
python bo.py --new_philly_jobs "philly_id_1 philly_id_2" bayesian/07e509b8
```

**To submit more jobs with new parameters predicted by Bayesian Search**
```
python bo.py --num_new_jobs 4 --num_rounds 5 bayesian/07e509b8
```

**To let the bo.py run in the background**
```
# for the first time
nohup python -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9001 bo.json > bo.log &
# resume a bo search
nohup python -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9001 bayesian/11853456/ > 11853456.log &
```

**To list background processes**
```
ps ax | grep bo.py
```

**To kill a process**
```
kill PID
or
pkill -f bo.py
```

