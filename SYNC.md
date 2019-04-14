# Sync with https://github.com/pytorch/fairseq

1. Check remote repository.

```
git remote -v
```

1. If upstream repo is not in the remote repo, for example, the following two lines are not shown:

upstream        https://github.com/pytorch/fairseq (fetch)
upstream        https://github.com/pytorch/fairseq (push)

Then add the upstream repo.

```
git remote add upstream https://github.com/pytorch/fairseq
```

1. Fetch the branches and their respective commits from the upstream repository. Commits to master will be stored in a local branch, upstream/master.

```
git fetch upstream
```

1. Check out your fork's local master branch.

```
git checkout master
```

1. Merge the changes from upstream/master into your local master branch.

```
git merge upstream/master
```

1. Push the changes.

```
git push
```
