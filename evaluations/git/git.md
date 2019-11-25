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

