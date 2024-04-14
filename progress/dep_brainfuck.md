a lot of shit had to be done, but it's pretty straightforward luckily

the conda_env file doesn't work at all and all attempts to fix it failed
i had to build stuff the good old way and install packages when needed. this almost worked, but then i ran into some difficulties with versions. i'm mostly fixing everything using the most straightforward methods (just installing different versions that were in the original conda env file)

when running with error with numpy, i tried downgrading it and then fixing the packages. this didn't work because packages kept installing the latest numpy version. so i instead installed additional packages first, and then downgraded numpy

i got it workinig. the updated conda config is in the conda_env_win.yaml file

i got another issue when pushing my changes to git. For the future, if you want to make some changes to the repo and copy it to your private account, do this:

```
git remote set-url origin 'your_url'
"make your commit like you would normally"
git push -u origin main
```

this should work

the conda env win yaml file doesn't work when i tried reinstalling the environment. says there is some pip subprocess error that requires python >= 3.9
the errors disappeared after i added the line `- --extra-index-url https://download.pytorch.org/whl/cu117` right below pip dependencies definitions in the yml file


TURSO BRAINFUCK

module load Python/3.8.6-GCCcore-10.2.0
mkdir python386
virtualenv python386
cd python386
source bin/activate
pip install -r ../requirements.txt
(error when installing backcall)

module load Anaconda3 worked
the environment installed successfully (wtf)

