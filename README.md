# shadow_removal

## how to build code
1. login to the hpc as normal and navigate to your $HOME/ece569 directory. 
    
    `$ cd ~/ece569`

    NOTE: this should be the same place your labs and build_dir are already located
    
1. clone or unzip the repo here in a folder titled shadow_removal

    `$ git clone https://github.com/mbarmstrong/shadow_removal.git`

1. navigate to build_dir and clean out everything

    `$ cd build_dir`
    
    `$ rm -r *`
    
    or if you don't have one yet
    
    `$ mkdir build_dir`
    
    `$ cd build_dir`

1. run the following commands to build

    `$ CC=gcc cmake3 ../shadow_removal/ -DCMAKE_BUILD_TYPE=Debug`
    
    `$ make`
 
 
 
## running the code on an interactive session

1. login to ocelote
2. (optional) add the isesh function to your .bashrc file from the steps below
3. launch the session with the folloiwng command

    `qsub -I -N add -W group_list=ece569 -q standard -l select=1:ncpus=2:mem=12gb:ngpus=1 -l walltime=00:5:00`
    
    or if you added the shortcut to your .bashrc
    
    `isesh 5`
   
   this will connect with a node on the hpc for 5 min (you can specify however many minutes you'd like) and now you can launch executables with GPU kernels directly from the command line.
  
1. to execute the shadow removal solution:

   `$ $HOME/ece569/build_dir/ShadowRemoval_Solution -i <path_to_image> -t image`
   
   or launch the shell script in the pbs scripts folder:
   
   `$ $HOME/ece569/shadow_removal/pbs_scripts/run_shadow_removal.sh`   



## running a unit test

there are unit tests associated with all of the phases of the shadow removal algoritm. Execute a unit test by running any of the following on an interactive session:

 `$ $HOME/ece569/shadow_removal/pbs_scripts/run_color_convert.sh`   
 `$ $HOME/ece569/shadow_removal/pbs_scripts/run_convolution.sh`   
 `$ $HOME/ece569/shadow_removal/pbs_scripts/run_erosion.sh`   
 `$ $HOME/ece569/shadow_removal/pbs_scripts/run_otsu_method.sh`   
 `$ $HOME/ece569/shadow_removal/pbs_scripts/run_result_integration.sh`   



## how to view timing output

1. navigate to output direcotry after running the shadow removal solution:

    `$ cd $HOME/ece569/shadow_removal/output`
    
2. open kernel kernel_times.csv in your favorite editor. for example:

    `$ gedit kernel_times.csv`

## how to debug and view output images

1. open globals.h file

    `$ cd $HOME/ece569/shadow_removal/src/globals.h`
    
2. find the PRINT_DEBUG flag and change it to 1

    `#define PRINT_DEBUG 1`
    
3. recompile the code

    `$ cd $HOME/ece569/build_dir/`

    `$ make`
    
4. run the shadow removal solution based on instructions above. Debug states will now be printed to the screen. In additon .ppm images of intermidiate steps and final image will appear in the output directory `$HOME/ece569/shadow_removal/output`



## usefull bash profile setup

here is Coale's .bashrc file, your equivalent .bashrc is located in your $HOME directory. to set this up:

1. go to home directory

    `$ cd ~`

1. open .bashrc (if not there, create a new one)

    `$ gedit .bashrc`

1. copy paste code in file and save

1. source the file

    `$ source .bashrc`
1. you now have all the settings loaded, every time you login from now on these settings will be automatically applied


```
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

OS=$(cat /etc/*release | grep CentOS | grep release | grep -o -E '[0-9]\.[0-9]{1,2}'| head -1)

echo "Host is running CentOS release $OS"

bind '"\e[A":history-search-backward'
bind '"\e[B":history-search-forward'

alias bd="cd /home/u29/cjcoopr/ece569/build_dir"

#ocelot setup
if [ "$OS" = "6.10" ]; then

  module load gcc
  module load cuda91/toolkit/9.1.85

  function isesh {

    echo "reqesting session for $1 min"
    qsub -I -N add -W group_list=ece569 -q standard -l select=1:ncpus=2:mem=12gb:ngpus=1 -l walltime=00:$1:00
  }

#elgato setup
elif [ "$OS" = "7.9" ]; then

  module load openmpi3
  module load cuda10

  function isesh {

    echo "reqesting session for $1 min"
    qsub -I -N add -W group_list=ece569 -q windfall -l select=1:ncpus=2:mem=12gb:ngpus=1 -l walltime=00:$1:00
  }
fi
```

