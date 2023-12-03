# CFDcoupling STARCCM
simili gym environment to be used in conjunction with the SandQVA/cfd_sureli repository to perform a DRL of flying flat plate using Star-CCM+

Requirements:
- Python 3.7 or older versions
- torch, torchvision
- imageio
- gym
- matplotlib
- PyYAML
- numpy

Usage :

* Setting the case
- Clone the SandQVA/cfd_sureli repository
- Inside this repository create a cfd folder
- Go to this new folder and clone .../starccm repository
- Set the config.yaml file contained in cfd_sureli/cfd/starccm/ according to the case you want to run

* Training
- run the RL algo --> python3 -u train 'AGENT' --appli='starccm' (add the --load and --loadrm arguments to restart a previous training)
- run the starccm simualtion .sim with the macro macro_externalfiles.java --> starccm -batch macro_externalfiles.java flatplate_coarse.sim
Be careful : !!! python has to be run before starccm !!!

* Testing
- run the RL algo --> python3 -u test 'AGENT' --appli='starccm' --f='training folder'
- run the starccm simualtion .sim with the macro macro_externalfiles.java --> starccm -batch macro_externalfiles.java flatplate_coarse.sim
Be careful : !!! python has to be run before starccm !!!
