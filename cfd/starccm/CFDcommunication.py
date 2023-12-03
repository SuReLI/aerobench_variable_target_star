# -*- coding: utf-8 -*-

import numpy as np
import time
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

class CFDcommunication:

    def __init__(self,config):
        
        self.config = config
       
        # initial and target conditions from config file
        self.xA = 0.
        self.yA = 0.
        self.uA = config["UA"]
        self.vA = config["VA"]

        self.B_array = []
        Btype=config['BTYPE']
        # B is fixed, values are given in the config file 
        if Btype=='fixed':
            self.xB = self.config["XB"]
            self.yB = self.config["YB"]
            self.B = np.array([self.xB,self.yB])
            self.B_array.append(self.B)
            A = np.array([self.xA,self.yA])
            self.rho0 = np.linalg.norm(A-self.B)
        # B is set randomly (in a given range) at each episode
        elif Btype=='random': 
            self.xB = 0.
            self.yB = 0. 
            self.B = np.array([self.xB,self.yB])
            self.rho0 = self.config["DISTANCE_RANGE"][1]
            if config['BTYPE_EVAL']=='fixed':
                self.xB = self.config["XB"]
                self.yB = self.config["YB"]
                self.B = np.array([self.xB,self.yB])
                self.B_array.append(self.B)
        else: print('Btype not correctly defined')
        self.B_batch = config["B_BATCH"]

        # some parameters
        self.nb_ep = 0
        self.done = False
        self.pitch = 0.
        self.pitchpy = 0.
        self.pitchrate = 0.
        self.max_pitchrate = config["MAX_PITCHRATE"]
        self.fx = 0.
        self.fy = 0.

        # state initialisation
        self.cartesian_init = np.array([self.xA, self.yA, self.uA, self.vA, self.pitch])
        self.state = np.zeros(6)

        # attributs needed by the rl code or gym wrappers
        self.action_space = collections.namedtuple('action_space', ['low', 'high', 'shape'])(-self.max_pitchrate, self.max_pitchrate, (1,))
        self.action_size = 1
        self.observation_space = collections.namedtuple('observation_space', ['shape'])(self.state.shape)
        self.reward_range = None
        self.metadata = None

        # dt_array and var_array initialisation (for postproc and plotting purposes only, not required by the application)    
        self.cfd_var_episode = []
        self.cfd_var_names = ['x', 'y', 'u', 'v', 'fx', 'fy']
        self.cfd_var_array = np.zeros([len(self.cfd_var_names), config["MAX_EPISODES"],config["MAX_STEPS"]+1])
        self.rl_var_episode = []
        self.rl_var_names = ['rho', 'sintheta', 'costheta', 'rhodot', 'thetadot', 'pitch', 'action', 'reward', 'pitchpy']
        self.rl_var_array = np.zeros([len(self.rl_var_names), config["MAX_EPISODES"], config["MAX_STEPS"]])


        #set of routes and files allowing coupling between STAR CCM+ and the RL code 
        self.startstoproute = 'cfd/starccm/startstop/'
        self.exporteddataroute = 'cfd/starccm/exporteddata/'
        
        self.finishsimulation = 'finishsimulation.txt'
        self.finishsimulationflag = 'finishsimulationflag.txt'
        
        self.pitchflag = 'pitchflag.txt'
        self.actiontoCFD = 'actiontoCFD.txt'
        
        self.resetsimulation = 'resetsimulation.txt'
        self.resetsimulationflag = 'resetsimulationflag.txt'
        
        self.stepdone = 'stepdone.txt'
        self.stepdoneflag = 'stepdoneflag.txt'

        self.filetranslationx = 'translationx.txt'
        self.filetranslationy = 'translationy.txt'
        self.filevelocityx = 'velocityx.txt'
        self.filevelocityy = 'velocityy.txt'
        self.fileaccelerationx = 'accelerationx.txt'
        self.fileaccelerationy = 'accelerationy.txt'
        self.filebetappi = 'betappi.txt'

        self.fileforcex = 'forcex.txt'
        self.fileforcey = 'forcey.txt'

	# initialise the files for the simulation
        self.clearfiles()
        self.initialisefiles()
        

    def step(self,action):
        old_polar_state = self.state
        self.pitchrate = action + np.random.normal(scale=self.max_pitchrate*self.config["ACTION_SIGMA"])
        # be careful to set the same values in config.yaml, macro_external.java and in the CFD.sim files
        self.pitchpy = self.pitchpy + self.pitchrate * self.config["DELTA_TIME"] * self.config["CFD_ITERATIONS"]

        # write the action computed by DDPG to STARCCM+
        self.sendactionCFD(self.pitchrate)
        
        fileroute = self.startstoproute
        
        filenameflag = self.stepdoneflag
        lookforflag = '0.0'
        writeflag = '1.0\n'
        
        filename = self.stepdone
        lookforvalue = '1'
        writevalue = '0'
        
        checkflag_writedataTXT(fileroute,filenameflag,lookforflag,writeflag, fileroute,filename,lookforvalue,writevalue)
        
        self.waitSTARCCM()
        
        new_cartesian_state = self.readstatesCFD()
        self.state = self.get_state_in_normalized_polar_coordinates(new_cartesian_state)
        #print('state', np.array2string(self.state, formatter={'float_kind':lambda x: "%.5f" % x}))
 
        # compute reward and check if the episode is over (done)
        reward = self.compute_reward(old_polar_state, self.state)
        won, lost = self.is_won_or_lost(self.state)
        if won or lost:
            self.done = True
        if self.done:
            reward = self.update_reward_if_done(reward, won, lost)
        done = self.done

        # save data for printing
        self.cfd_var_episode = self.cfd_var_episode + [list(new_cartesian_state) + [self.fx] + [self.fy]]
        self.rl_var_episode = self.rl_var_episode + [list(self.state) + [action] + [reward] + [self.pitchpy]]
        
        return [self.state.copy(), reward, done, None]


    def reset(self, Btype='random'):
        self.nb_ep +=1
        self.rl_var_episode = []
        self.done = False
        self.pitch = 0.
        self.pitchpy = 0.
        self.fx = 0.
        self.fy = 0.

        # B is fixed, values are given in the config file 
        if Btype=='fixed':
            pass # values defined in the class init

        # B is set randomly (in a given range) at each episode
        elif Btype=='random':
            self.update_B_random()
            self.B = np.array([self.xB,self.yB])
            self.B_array.append(self.B)

        # a fixed set of B positions is used
        elif Btype=='batch':
            self.xB = self.B_batch[(self.nb_ep-1) % len(self.B_batch)][0]
            self.yB = self.B_batch[(self.nb_ep-1)% len(self.B_batch)][1]
            self.B = np.array([self.xB,self.yB])
            self.B_array.append(self.B)
            #print('B coordinates = ', self.B)

        else:
            print('B not correctly defined')

        # define initial state according to the position of B
        self.state = self.get_state_in_normalized_polar_coordinates(self.cartesian_init)
        # fill array with initial state
        self.cfd_var_episode = [list(self.cartesian_init) + [self.fx] + [self.fy]]

        self.clearfiles()
        fileroute = self.startstoproute
        fileresetflag = self.resetsimulationflag
        lookforflag = ''
        writeflag = '1.0\n'
        
        filereset = self.resetsimulation
        lookforvalue = ''
        writevalue = '1'
        
        checkflag_writedataTXT(fileroute,fileresetflag,lookforflag,writeflag, fileroute,filereset,lookforvalue,writevalue)
        
        return self.state.copy()


    # defined in order to mimic a gym environment
    def render(self, mode='human'):
        pass


    # defined in order to mimic a gym environment
    def close(self):
        pass


    # defined in order to mimic a gym environment
    def seed(self):
        pass


    def compute_reward(self, old_polar_state, new_polar_state):
        if self.config["REWARD_TYPE"] == 'dense':
            delta_rho = new_polar_state[0] - old_polar_state[0]
            reward = -100*delta_rho
        elif self.config["REWARD_TYPE"] == 'sparse':
            reward = 0.0
        else:
            print('!!! please define reward !!!')

        return reward


    def update_reward_if_done(self, reward, won, lost):
        if won: reward += 10.
        elif lost: reward += -10.

        return reward

    
    def is_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]) <= 10**(-2):
            won = True
        elif polar_state[2] <= 0.:
            lost = True

        return won, lost


    def print_won_or_lost(self, polar_state):
        won = False
        lost = False

        if np.abs(polar_state[0]) <= 10**(-2):
            won = True
            print('won')
        elif polar_state[2] <= 0.:
            lost = True
            print('lost')
        else: print('lost')

        return won, lost


    # update B coordinates as required in the CFD config file
    def update_B_random(self):
        env_angle_range = np.array(self.config["ANGLE_RANGE"])/180*np.pi
        env_distance_range = np.array(self.config["DISTANCE_RANGE"])
        thetaA = -np.random.uniform(env_angle_range[0], env_angle_range[1])
        rhoAB = np.random.uniform(env_distance_range[0], env_distance_range[1])

        self.xB = rhoAB * np.cos(np.pi+thetaA)
        self.yB = rhoAB * np.sin(np.pi+thetaA)


# Coupling utility functions ------------------------------------------

    def readstatesCFD(self):
        
        translationx = readTXT(self.exporteddataroute,self.filetranslationx)
        translationy = readTXT(self.exporteddataroute,self.filetranslationy)
        velocityx = readTXT(self.exporteddataroute,self.filevelocityx)
        velocityy = readTXT(self.exporteddataroute,self.filevelocityy)        
        betappi = readTXT(self.exporteddataroute,self.filebetappi)

        self.fx = float(readTXT3(self.exporteddataroute,self.fileforcex)[0])
        self.fy = float(readTXT3(self.exporteddataroute,self.fileforcey)[1])

        newstatecartesian = [translationx, translationy, velocityx, velocityy, betappi]
        return newstatecartesian


    def finishCFD(self,done=False):
        #done: boolean that takes the values
        # True -> finish the simulation
        # False -> not finish the simulation, just update the value of the flag
        fileroute = self.startstoproute
        filefinish = self.finishsimulation
        filefinishflag = self.finishsimulationflag
        filestepdoneflag = self.stepdoneflag
        
        flag = ''
        while '0.0' not in flag:
            with open(fileroute+filefinishflag,'r+') as f:
                flag = f.read()
                time.sleep(0.05)
            f.close()

        #in fact, not needed condition since it would not have passed the while loop    
        if '0.0' in flag:
            #update the flag to permit continue to STAR CCM+
            flag = '1.0\n'
            if done==True:
                with open(fileroute+filefinish,'r+') as f2:
                    f2.seek(0)
                    f2.write('1')
                    f2.truncate()      
#                   #update flag for finishsimulationflag
#                   flag = '1.0\n'
                    with open(fileroute+filefinishflag,'r+') as f:
                        f.seek(0)
                        f.write(flag)
                        f.truncate()
                    with open(fileroute+filestepdoneflag,'r+') as f3:
                        f3.seek(0)
                        f3.write(flag)
                        f3.truncate()
                    f3.close()
                f2.close()
            else:
#                #update the flag to permit continue to STAR CCM+
#                flag = '1.0\n'
                with open(fileroute+filefinishflag,'r+') as f:
                    f.seek(0)
                    f.write(flag)
                    f.truncate()
                f.close()
            
            
    def clearfiles(self):
        
        clearTXT(self.exporteddataroute,self.actiontoCFD)
        clearTXT(self.exporteddataroute,self.filetranslationx)
        clearTXT(self.exporteddataroute,self.filetranslationy)
        clearTXT(self.exporteddataroute,self.filevelocityx)
        clearTXT(self.exporteddataroute,self.filevelocityy)
        clearTXT(self.exporteddataroute,self.fileaccelerationx)
        clearTXT(self.exporteddataroute,self.fileaccelerationy)
        clearTXT(self.exporteddataroute,self.filebetappi)


    def initialisefiles(self):
        #check and set as not step done in STARCCM+ simulation
        datastr = '1'
        writeTXT(self.startstoproute,self.stepdone,datastr)
        
        datastr = '0.0\n'
        writeTXT(self.startstoproute,self.stepdoneflag,datastr)
                
        #check and set as not to finish the STARCCM+ simulation
        datastr = '0'
        writeTXT(self.startstoproute,self.finishsimulation,datastr)
        
        datastr = '0.0\n'
        writeTXT(self.startstoproute,self.finishsimulationflag,datastr)

        #check and set as reset for the STARCCM+ simulation
        datastr = '0.0\n'
        writeTXT(self.startstoproute,self.resetsimulation,datastr)

        datastr = '0.0\n'
        writeTXT(self.startstoproute,self.resetsimulationflag,datastr)
        
        #check no pitch rate has been computed
        datastr = '0.0\n'
        writeTXT(self.startstoproute,self.pitchflag,datastr)
        

    # export the action information to external .csv file to be read by STARCCM
    def sendactionCFD(self,action):
        routeflag = self.startstoproute
        fileflag = self.pitchflag
        lookforflag = '0.0'
        writeflag = '1.0\n'
        
        routeexchangefiles = self.exporteddataroute
        actionfile = self.actiontoCFD
        #not looked for value, so it is left 'empty'
        lookforvalue = ''
        writevalue = str(action[0])
        
        checkflag_writedataTXT(routeflag,fileflag,lookforflag,writeflag, routeexchangefiles,actionfile,lookforvalue,writevalue)
    
    
    # wait until complete iteration of STARCCM+
    # continuously read a file to check if it has written on it '1', when it has written '1'
    #it means that the step in STARCCM+ is over
    def waitSTARCCM(self):
    
        fileroute = self.startstoproute
        filenameflag = self.stepdoneflag
        fileresetflag = self.resetsimulationflag
    
        #check that the turn of python has arrived to continue with the evaluation
        flag = ''
        while '0.0' not in flag:
            with open(fileroute+filenameflag,'r+') as f:
                flag = f.read()
                time.sleep(0.05)
            f.close()
    
        #change the flag of the reset file too, not only the one of the step to keep
        #on with the shift of DDPG and STAR CCM+ evaluation
        resetflag = ''
        while '0.0' not in resetflag:
            with open(fileroute+fileresetflag,'r+') as f:
                resetflag = f.read()
                #print(flag + "\n")
                time.sleep(0.05)
            f.close()
        #in fact, not needed condition since it would not have passed the while loop
        if '0.0' in resetflag:
            with open(fileroute + fileresetflag, 'r+') as f:
                f.seek(0)
                f.write('1.0\n')
                f.truncate()
            f.close()



# Below are only utility functions ------------------------------------------

    def get_state_in_normalized_polar_coordinates(self, cartesian_state):
        BP = cartesian_state[0:2]-self.B
        rho = np.linalg.norm(BP)
        theta = np.arctan2(BP[1],BP[0])

        u = cartesian_state[2]
        v = cartesian_state[3]
        pitch = - (np.pi - cartesian_state[4])

        rhoDot = u * np.cos(theta) + v * np.sin(theta)
        thetaDot = - u * np.sin(theta) + v * np.cos(theta)
        polar_state = np.array([rho, np.sin(theta), np.cos(theta), rhoDot, thetaDot, pitch])
        normalized_polar_state = self.normalize_polar_state(polar_state)

        return normalized_polar_state


    def normalize_polar_state(self, state):
        normalized_state = np.zeros(6)
        normalized_state[0] = state[0]/self.rho0
        normalized_state[1] = state[1]
        normalized_state[2] = state[2]
        normalized_state[3] = state[3]/np.sqrt(self.uA**2+self.vA**2)
        normalized_state[4] = state[4]/(self.max_pitchrate/100)
        normalized_state[5] = state[5]

        return normalized_state


    def fill_array_tobesaved(self):
        for i in range(len(self.cfd_var_names)):
            for k in range(len(self.cfd_var_episode)):
                self.cfd_var_array[i, self.nb_ep-1, k] = self.cfd_var_episode[k][i]

        for i in range(len(self.rl_var_names)):
            for k in range(len(self.rl_var_episode)):
                self.rl_var_array[i, self.nb_ep-1, k] = self.rl_var_episode[k][i]


    def print_array_in_files(self, folder):
        for i, var in enumerate(self.cfd_var_names):
            filename = folder+'/'+var+'.csv'
            np.savetxt(filename, self.cfd_var_array[i,:,:], delimiter=";")

        for i, var in enumerate(self.rl_var_names):
            filename = folder+'/'+var+'.csv'
            np.savetxt(filename, self.rl_var_array[i,:,:], delimiter=";")

        filename = folder+'/Bcoordinates.csv'
        np.savetxt(filename, self.B_array, delimiter=";")


    def plot_training_output(self, returns, eval_returns, freq_eval, folder):
        xlast = np.trim_zeros(self.cfd_var_array[0,-1,:], 'b')
        ylast = self.cfd_var_array[1,-1,:len(xlast)] 

        cumulative_reward = self.rl_var_array[7,:,:].sum(axis=1)

        best = np.argmax(cumulative_reward)
        xbest = np.trim_zeros(self.cfd_var_array[0,best,:], 'b')
        ybest = self.cfd_var_array[1,best,:len(xbest)]
        if len(self.B_array) == 1:
            Bbest = self.B_array[0]
        else: 
            Bbest = self.B_array[best]

        ep_eval = [i*freq_eval for i in range(1, len(eval_returns)+1)]

        plt.cla()
        plt.figure(figsize=(14, 5))
        plt.tight_layout()
        plt.suptitle(folder.rsplit('/', 1)[1])
        plt.subplot(1,3,2)
        plt.title('Trajectories')
        plt.plot([self.xA, self.xB], [self.yA, self.yB], color='black', ls='--', label='Ideal path')
        plt.plot(xlast, ylast, color='black', label='last path ep='+str(self.config['MAX_EPISODES']))
        plt.plot([self.xA, Bbest[0]], [self.yA, Bbest[1]], color='green', ls='--', label='Ideal path')
        plt.plot(xbest, ybest, color='green', label='best path ep='+str(best))
        plt.grid()
        #plt.axis('equal')
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)
        plt.legend(fontsize = 10, loc='best')

        plt.subplot(1,3,1)
        plt.title('Training and evaluation returns')
        plt.plot(returns, color='black', label='training')
        plt.plot(ep_eval, eval_returns, color='red', label='evaluation')
        plt.grid()
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Return', fontsize=14)
        plt.legend(loc=0)

        plt.subplot(1,3,3)
        plt.plot(np.transpose(self.B_array)[0], np.transpose(self.B_array)[1], '.', color='blue')
        if self.config["BTYPE_EVAL"] == "batch":
            plt.plot(np.transpose(self.B_batch)[0], np.transpose(self.B_batch)[1], '.', color='red')
        plt.scatter(self.xA, self.yA, 150, color='black', zorder=1.0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'{folder}/train_output.png')


    def plot_testing_output(self, returns, folder):
        score = sum(returns)/len(returns) if returns else 0
        xlast = np.trim_zeros(self.cfd_var_array[0,0,:], 'b')
        ylast = self.cfd_var_array[1,0,:len(xlast)]

        plt.cla()
        plt.figure(figsize=(10, 5))
        plt.suptitle(folder.rsplit('/', 2)[1])
        plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.90, hspace = 0.6)

        plt.subplot(1,3,1)
        plt.title('Testing return for each episode')
        plt.plot(returns, 'o-', color='black', label='training')
        plt.grid()
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Return', fontsize=14)

        plt.subplot(1,3,2)
        n_cmap = len(returns)
        cmap = pl.cm.tab10(np.linspace(0,1,n_cmap))

        if len(self.B_array) == 1:
            plt.plot([self.xA,self.B_array[0][0]], [self.yA,self.B_array[0][1]], color='black', ls='--', label='Ideal path')
        else:
            for i in range(len(returns)):
                plt.plot([self.xA,self.B_array[i][0]], [self.yA,self.B_array[i][1]], color=cmap[i], ls='--', label='Ideal path')
        
        for i in range(len(returns)):
            x = np.trim_zeros(self.cfd_var_array[0,i,:], 'b')
            y = self.cfd_var_array[1,i,:len(x)]
            plt.plot(x,y, color=cmap[i], label='test path')
        plt.grid()
        plt.axis('equal')
        plt.xlabel('x (m)', fontsize=14)
        plt.ylabel('y (m)', fontsize=14)

        plt.subplot(1,3,3)
        plt.plot(np.transpose(self.B_array[:len(returns)])[0], np.transpose(self.B_array[:len(returns)])[1], '.', color='red')
        plt.scatter(self.xA, self.yA, 150, color='black', zorder=1.0)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(f'{folder}/test_output.png')


def readTXT(fileroute,filename):
    readdata=[]
    try:
        with open(fileroute+filename,'r+') as f:
            readdata = f.read()
        data = float(readdata[:-1])
    finally:
        f.close()
    return data


def readTXT3(fileroute,filename):
    readdata=[]
    try:
        with open(fileroute+filename,'r+') as f:
            readdata = f.read()
        data = readdata.replace('[', '').split(',')
    finally:
        f.close()
    return data
 
 
def writeTXT(fileroute,filename,datastr):
    try:
        with open(fileroute+filename,'r+') as f:
            f.seek(0)
            f.write(datastr)
            f.truncate()
    finally:
        f.close()
    
    
def clearTXT(fileroute,filename):
    try:
        with open(fileroute+filename,'r+') as f:
            f.seek(0)
            f.truncate()
    finally:
        f.close()        


#function that checks the semaphore to write the new values on the corresponding files
def checkflag_writedataTXT(routeflag,fileflag,lookforflag,writeflag, routefile,file,lookforvalue,writevalue):
    #wait until the condition of the flag that is given as input is satisfied
    #for this problem, the two values of the flag are:
    #0 -> only Python can change the file
    #1 -> only Java can change the file
    flag = ''
    while lookforflag not in flag:
        with open(routeflag+fileflag,'r+') as f:
            flag=f.read()
            #delay to avoid failure in files because of open and close at the 'same time'
            time.sleep(0.05)
        f.close()
        
    #in fact, not needed condition since it would not have passed the while loop
    if lookforflag in flag:
        with open(routefile+file,'r+') as f2:
            readvalue = f2.read()
            #check that the condition that is wanted in the file is fulfilled and
            #update the value according to the requirements
            if lookforvalue in readvalue:
                value = writevalue
                f2.seek(0)
                f2.write(value)
                f2.truncate()
                #change the value of the flag so that STAR CCM+ can access to the file
                with open(routeflag+fileflag,'r+') as f:
                    f.seek(0)
                    f.write(writeflag)
                    f.truncate()
                f.close()
        f2.close()


