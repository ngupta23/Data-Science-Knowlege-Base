# Debugging Code Merge Conflicts

<<<<<<< HEAD

-- Content from local branch --

=======  

-- Content from Remote --

\>>>>>>> 6213be7b31803862f4260a1f6d549fc150177d34

**Explanation**

* Content between <<<<<<< HEAD & ======= is the content on the local branch.
* Content between ======= HEAD & >>>>>>> 6213be7b31803862f4260a1f6d549fc150177d34 is the content from remote


# Deleting a branch
* https://stackoverflow.com/questions/2003505/how-do-i-delete-a-git-branch-locally-and-remotely Check out Executive Summary
  
  ```
  # For remote repo (with example)
  $ git push -d <remote_name> <branch_name>
  $ git push -d origin <branch_name>
  
  # For local branch
  $ git branch -d <branch_name>
  ```

# Pruning a local branch that has been deleted remotely
* https://stackoverflow.com/questions/13064613/how-to-prune-local-tracking-branches-that-do-not-exist-on-remote-anymore

  ```
  $ git fetch -p
  ```

# Listing all branches
* https://stackoverflow.com/questions/12370714/git-how-do-i-list-only-local-branches

```
$ git branch -a # shows both remote and local branches.
$ git branch -r # shows remote branches.
$ git branch    # With no arguments, existing branches are listed and the current branch will be highlighted with an asterisk.
```

