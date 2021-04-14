# shadow_removal

## How to build code
1. login to the hpc as normal and navigate to your $HOME/ece569 directory
    
    `$ cd ~/ece569`

    NOTE: this should be the same place your labs and build_dir are already located
    
1. clone the repo

    `$ git clone https://github.com/mbarmstrong/shadow_removal.git`

1. navigate to build_dir and clean out everything

    `$ cd build_dir`
    
    `$ rm -r *`

1. run the following commands to build

    `$ CC=gcc cmake3 ../shadow_removal/ -DCMAKE_BUILD_TYPE=Debug`
    
    `$ make`
   
## Usefull bash profile setup

Here is Coale's .bashrc file. Your equivalent .bashrc is located in your $HOME directory. I think everyone will find the iteractive session part usefull. To set this up:

1. go to home directory

    `$ cd ~`

1. open .bashrc (if not there, create a new one)

    `$ gedit .bashrc`

1. copy paste code in file and save

1. source the file

    `$ source .bashrc`
1. you now have all the settings loaded, every time you login from now on these settings will be automatically applied.

```
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# User specific aliases and functions
module load gcc
module load cuda91/toolkit/9.1.85

bind '"\e[A":history-search-backward'
bind '"\e[B":history-search-forward'

alias bd="cd /home/u29/cjcoopr/ece569/build_dir"
function isesh {

    echo "reqesting session for $1 min"
    qsub -I -N add -W group_list=ece569 -q windfall -l select=1:ncpus=2:mem=12gb:ngpus=1 -l walltime=00:$1:00
}

```

## Running an interactive session

1. first add the isesh function to your .bashrc file from the steps above
1. login to ocelote or elgato
1. launch the session with the folloiwng command

    `$ isesh 5`
   
   This will connect with a node on the hpc for 5 min but you can specify however much time you'd like. Here you can now launch executables with GPU kernels directly for the command line. For example:
   
   `$ $HOME/ece569/build_dir/ShadowRemoval_Solution -i <path_to_image> -t image`
