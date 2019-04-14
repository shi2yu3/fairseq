# Sync with https://github.com/pytorch/fairseq

## Check remote repository.

```
git remote -v
```

## If upstream repo is not in the remote repo, for example, the following two lines are not shown:

```
upstream        https://github.com/pytorch/fairseq (fetch)
upstream        https://github.com/pytorch/fairseq (push)
```

Then add the upstream repo.

```
git remote add upstream https://github.com/pytorch/fairseq
```

## Fetch the branches and their respective commits from the upstream repository. Commits to master will be stored in a local branch, upstream/master.

```
git fetch upstream
```

## Check out your fork's local master branch.

```
git checkout master
```

## Merge the changes from upstream/master into your local master branch.

```
git merge upstream/master
```

## Push the changes.

```
git push
```
