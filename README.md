# shadow_removal

1. login to the hpc as normal and navigate to your $HOME/ece569 directory
    NOTE: this should be the same place your labs and build_dir are already located
    
1. clone the repo
`$ git clone https://github.com/mbarmstrong/shadow_removal.git`

1. navigate to build_dir and clean out everything

1. run the following commands to build

    `$ CC=gcc cmake3 ../shadow_removal/ -DCMAKE_BUILD_TYPE=Debug`
    
    `$ make`
   
