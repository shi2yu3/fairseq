# Bayesian search on top of Philly

**First run**

```
python bo.py --num_new_jobs 4 bo.json
```

It will generate a folder with job json and job info, say bayesian/07e509b8, and submit 4 jobs to Philly.

**To track and update the jobs**

```
python bo.py --num_new_jobs 0 bayesian/07e509b8
```

**To submit more jobs with new parameters predicted by Bayesian Search**

```
python bo.py --num_new_jobs 2 bayesian/07e509b8
```
