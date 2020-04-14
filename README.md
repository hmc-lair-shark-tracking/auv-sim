# auv-sim
## Clone this repository
You can clone (download the auv-sim repository on your computer) by doing the following
1. Go to the directory where you want to download this repo
2. In your terminal, `git clone https://github.com/hmc-lair-shark-tracking/auv-sim.git`

For more information, see: 
https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

## Create your own branch 
Once you clone the code, you should create your own branch before you start making changes. Branches allow 
each of us to have our own place to work on the code without affecting other people's code. 

For more information about branches, see: 
https://guides.github.com/introduction/flow/

This website gives a general overview of how branches and Github work. 

You can create a branch locally by doing the following: 
1. In your terminal, go to this repository's directory
(When you type `ls`, you should see the robotSim.py file)

2. Type `git checkout -b [your-branch-name]`
(For example `git checkout -b trackTrajectory`)
Your terminal should print out "Switched to a new branch 'your-branch-name'"

## Add -> Commit -> Push: How to keep track of your changes
Once you finish editting the code, and you want to keep track of the changes you have made, you should do 
the following: 

1. In your terminal, go to this repository's directory
(When you type `ls`, you should see the robotSim.py file)

2. Type `git status`
You should be able to see:
```
On branch [your-branch-name]
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)
  
        modified:   robotSim.py
```
3. Type  `git add [file-name]` (ex. `git add robotSim.py`) if you want to keep track of the changes done to a 
specific file

(or you can type `git add --a` if you want to keep track of all the changes done to any files)

4. Now, if you type `git status` again, you should see:
```
On branch [your-branch-name]
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        modified:   robotSim.py
```
5. To actually "store" your changes, you have to **commit**. 
Type `git commit -m "brief description of what you have done"` 
(ex. `git commit -m "add comments for trackTrajectory function`)

Now, your changes have been stored locally. 

6. However, if you want your changes to appear on Github and let other people see it, you have to **push**.
Usually, type `git push`

However, **if this is your first time pushing from this branch**, you need to type:

`git push --set-upstream origin [your-branch-name]`

(You can technically still type `git push`, and git will helpfully tell you the right command to type.)

This is just a basic introduction of the basic Github work flow. For more information, see here:
https://guides.github.com/introduction/git-handbook/

## Install necessary libraries
Because our python code uses Matplotlib to show the plot, you need to make sure that you have installed
this library on your computer. 

In your terminal, run the following: 
```
python3 -m pip install -U pip
python3 -m pip install -U matplotlib
```

For more information, see:
https://matplotlib.org/3.2.1/users/installing.html

## Running the code
1. Go to the auv-sim folder, which contains robotSim.py
2. In your terminal, `python3 robotSim.py`
