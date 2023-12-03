// STAR-CCM+ macro: macro_externalfiles.java
// Written by STAR-CCM+ 12.06.011
package macro;
//package com.admfactory.io.socket;


import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.base.report.*;
import star.flow.*;
import star.sixdof.*;

import java.io.*;
import java.net.*;
import java.util.concurrent.TimeUnit;

import star.vis.*;

public class macro_externalfiles extends StarMacro {

	int iteration = 0;
	int iterationdone = 0;
	int nbsteps = 1;  // to set equal to CFD_ITERATIONS in the config file
	Simulation theSim;

	//variable to stop the evaluation of the simulation because DDPG has stopped
	int finished = 0;
	
	
	public void execute() {
		
		//disableupdatescene();

		String start_trigger;
		start_trigger = "startstop/stepdoneflag.txt";		
		String trigger = "";
		
		trigger = read_txt(start_trigger);
		while(Double.parseDouble(trigger) == 0){
			trigger = read_txt(start_trigger);
		}

		
		while(finished != 1){			
			
		    	theSim = getActiveSimulation();
			resetSimulation(theSim);
			
		    	// RUN and export the reports
			iterationdone = simulationStep(theSim, nbsteps); 
			
			if(iterationdone == 1){
				iteration += 1;
				// SAVE THE SIMULATION with corresponding backups
				//saveSimulation(theSim, iteration);
			}
			finished = finishsimulation(theSim);
			
		}
	}


	// STEP the simulation and notify to DDPG it is already finished
	private int simulationStep(Simulation theSim, int nbsteps){

		String filestepflag;
		filestepflag = "startstop/stepdoneflag.txt";		
		String flagfinish = "";
		
		String filecheckfinish;
		filecheckfinish = "startstop/finishsimulation.txt";
		String checkflagfinish = "";
		
		int iterated = 0; 
		String filestep;
		filestep = "startstop/stepdone.txt";
		String valuestrstep = "";
		
		flagfinish = read_txt(filestepflag);
		checkflagfinish = read_txt(filecheckfinish);
		
		while(Double.parseDouble(flagfinish)==0 && Double.parseDouble(checkflagfinish)==0){
			flagfinish = read_txt(filestepflag);
			checkflagfinish = read_txt(filecheckfinish);
			try{
				TimeUnit.MILLISECONDS.sleep(100);
			} catch(Exception e){}
		}
		if(flagfinish != null && !flagfinish.isEmpty() && Double.parseDouble(checkflagfinish)==0){
			if(Double.parseDouble(flagfinish)==1){
				valuestrstep = read_txt(filestep);

				if(valuestrstep != null && !valuestrstep.isEmpty()){
					if(Double.parseDouble(valuestrstep)==0){
						//change the pitch rate in the simulation
						pitchRateChange(theSim);
						//run step
						theSim.getSimulationIterator().step(nbsteps);
						// export the reports
						exportReports(theSim);
						//notify to DDPG that the iteration is performed
						write_txt(filestep, 1);
						write_txt(filestepflag, 0);
						iterated = 1;	
					}
				}
			}
		}
		return iterated;
	}


	// EXPORT REPORTS to .txt
	private void exportReports(Simulation theSim){

		LinearAccelerationReport acceleration1 = ((LinearAccelerationReport) theSim.getReportManager().getReport("6-DOF Body Acceleration 1"));
		double accelerationx = acceleration1.getValue();
		//acceleration1.printReport();
		
		LinearAccelerationReport acceleration2 = ((LinearAccelerationReport) theSim.getReportManager().getReport("6-DOF Body Acceleration 2"));
		double accelerationy = acceleration2.getValue();
    		//acceleration2.printReport();

    		TranslationReport translation1 = ((TranslationReport) theSim.getReportManager().getReport("6-DOF Body Translation 1"));
		double translationx = translation1.getValue();
    		//translation1.printReport();

    		TranslationReport translation2 = ((TranslationReport) theSim.getReportManager().getReport("6-DOF Body Translation 2"));
		double translationy = translation2.getValue();
    		//translation2.printReport();

    		LinearVelocityReport velocity1 = ((LinearVelocityReport) theSim.getReportManager().getReport("6-DOF Body Velocity 1"));
		double velocityx = velocity1.getValue();
    		//velocity1.printReport();

    		LinearVelocityReport velocity2 = ((LinearVelocityReport) theSim.getReportManager().getReport("6-DOF Body Velocity 2"));
		double velocityy = velocity2.getValue();
    		//velocity2.printReport();
			
    		ForceReport force1 = ((ForceReport) theSim.getReportManager().getReport("Force x"));
		Vector3 forcex = force1.getValue();
    		//force1.printReport();

    		ForceReport force2 = ((ForceReport) theSim.getReportManager().getReport("Force y"));
		Vector3 forcey = force2.getValue();
    		//force2.printReport();
    		
                MaxReport pitch1 = ((MaxReport) theSim.getReportManager().getReport("pitch_angle_pluspi"));
		double betappi = pitch1.getValue();
		
		
	    // LINUX
		// WRITE THE READ INFORMATION OF REPORTS IN A .txt file
		String accelerationxfilename = "exporteddata/accelerationx.txt";
		write_txt(accelerationxfilename, accelerationx);
		String accelerationyfilename = "exporteddata/accelerationy.txt";
		write_txt(accelerationyfilename, accelerationy);
		String velocityxfilename = "exporteddata/velocityx.txt";
		write_txt(velocityxfilename, velocityx);
		String velocityyfilename = "exporteddata/velocityy.txt";
		write_txt(velocityyfilename, velocityy);
		String translationxfilename = "exporteddata/translationx.txt";
		write_txt(translationxfilename, translationx);
		String translationyfilename = "exporteddata/translationy.txt";
		write_txt(translationyfilename, translationy);
		String betappifilename = "exporteddata/betappi.txt";
		write_txt(betappifilename, betappi);
		
		String forcexfilename = "exporteddata/forcex.txt";
		write_txt3(forcexfilename, forcex);
		String forceyfilename = "exporteddata/forcey.txt";
		write_txt3(forceyfilename, forcey);
		
		
	}

    	// SAVE THE SIMULATION MAKING 3 DIFFERENT COPIES TO AVOID FAILURE DURING SAVING
	private void saveSimulation(Simulation theSim, int iteration){
		int autosavenumb;
		autosavenumb = 0;
		if (iteration % 3 == 0){
		autosavenumb = 1;
		} else if (iteration % 3 == 1){
		autosavenumb = 2;
		} else if (iteration % 3 == 2){
		autosavenumb = 3;
		}
		// save the simulation
		theSim.saveState("autosave" + autosavenumb + ".sim");
	}


	//RESET THE SIMULATION when DDPG requests it
	private void resetSimulation(Simulation theSim){

		String fileresetflag;
		fileresetflag = "startstop/resetsimulationflag.txt";		
		String flag = "";
		
		String filereset;
		filereset = "startstop/resetsimulation.txt";
		String resetstr = "";
		
		flag = read_txt(fileresetflag);
		//wait until permission for entering the .txt file
		while(Double.parseDouble(flag)!=1){
			flag = read_txt(fileresetflag);
			try{
				TimeUnit.MILLISECONDS.sleep(100);
			} catch(Exception e){}
		}
		
		if(flag != null && !flag.isEmpty()){
			if(Double.parseDouble(flag)==1){
				
				resetstr = read_txt(filereset);
				
				if(resetstr != null && !resetstr.isEmpty()){
					if(Double.parseDouble(resetstr)==1){
						theSim.clearSolution();
						theSim.initializeSolution();
						/* // find pitch rate in simulation to modify
						UserFieldFunction pitch_rate = ((UserFieldFunction) theSim.getFieldFunctionManager().getFunction("Picth"));
						// modify pitch rate in simulation
						pitch_rate.setDefinition("0"); */
						
						write_txt(filereset, 0);
						
		
					}
				}
				write_txt(fileresetflag, 0);
			}
		}
	}

	private int finishsimulation(Simulation theSim){
		
		int finishsim = 0;
		
		String filefinishflag;
		//filefinishflag = "startstop\\finishsimulationflag.txt";
		filefinishflag = "startstop/finishsimulationflag.txt";
		String flagfinish = "";
		
		String filefinish;
		//filefinish = "startstop\\finishsimulation.txt";
		filefinish = "startstop/finishsimulation.txt";		
		String finishedstr = "";
		
		String filestepflag;
		//filestepflag = "startstop\\stepdoneflag.txt";
		filestepflag = "startstop/stepdoneflag.txt";
		String stepdoneflag = "";
		
		flagfinish = read_txt(filefinishflag);
		
		
		while(Double.parseDouble(flagfinish)!=1){
			flagfinish = read_txt(filefinishflag);
			try{
				TimeUnit.MILLISECONDS.sleep(100);
			} catch(Exception e){}
		}
		
		if(flagfinish != null && !flagfinish.isEmpty()){
			if(Double.parseDouble(flagfinish)==1){
				
				finishedstr = read_txt(filefinish);
				
				if(finishedstr != null && !finishedstr.isEmpty()){
					if(Double.parseDouble(finishedstr)==1){
						//clear the solution
						theSim.clearSolution();
						//Finish simulation
						write_txt(filefinish, 0);
						finishsim = 1;
					}
				}
				write_txt(filefinishflag, 0);
			}
		}
		
		stepdoneflag = read_txt(filestepflag);
		finishedstr = read_txt(filefinish);
		while(Double.parseDouble(stepdoneflag)==0 && finishsim==0){
			
			stepdoneflag = read_txt(filestepflag);
			finishedstr = read_txt(filefinish);
			try{
				TimeUnit.MILLISECONDS.sleep(100);
			} catch(Exception e){}
		}
		
		return finishsim;
	}

	//CHANGE PITCH RATE VALUE according to stated by DDPG
	private void pitchRateChange(Simulation theSim){
		
		String filepitchflag;
		filepitchflag = "startstop/pitchflag.txt";
		String flagpitch = "";
		flagpitch = read_txt(filepitchflag);
		
		String filepitch;
		filepitch = "exporteddata/actiontoCFD.txt";		
		String pitchstr = "";

		while(Double.parseDouble(flagpitch)!=1){
			flagpitch = read_txt(filepitchflag);
			try{
				TimeUnit.MILLISECONDS.sleep(100);
			} catch(Exception e){}
		}

		if(flagpitch != null && !flagpitch.isEmpty()){
			if(Double.parseDouble(flagpitch)==1){

				pitchstr = read_txt(filepitch);
				
				if(pitchstr != null && !pitchstr.isEmpty()){
					//System.out.println("pitch rate change");
					// find pitch rate in simulation to modify
					UserFieldFunction pitch_rate = ((UserFieldFunction) theSim.getFieldFunctionManager().getFunction("Picth"));
					// modify pitch rate in simulation
					pitch_rate.setDefinition(pitchstr);
					//update the value of the flag for the IA
					write_txt(filepitchflag, 0);
					
				}
			}
		}
	}

	//READ .txt file to provide information
	private String read_txt(String filename){
		String valuestr = "";
		try{
			File f = new File(resolvePath(filename));
			Scanner scanner = new Scanner(f);
			scanner.useDelimiter(";");
			while(scanner.hasNext()){
				valuestr = scanner.next();
			}
			scanner.close();
		} catch(Exception e){}
		return valuestr;
	}

	// WRITE IN THE TEXT FILE SOME NEW VALUES
	private void write_txt(String filename, double data) {

		try {
			// FileWriter(filepath, boolean).
			// boolean = true -> appending information
			// boolean = false -> overwriting information 
			FileWriter fw = new FileWriter(resolvePath(filename),false); 
			fw.write(data + "\r\n");
			fw.close();

		} catch(Exception e){}
	}


	// WRITE IN THE TEXT FILE SOME NEW VALUES
	private void write_txt3(String filename, Vector3 data) {

		try {
			// FileWriter(filepath, boolean).
			// boolean = true -> appending information
			// boolean = false -> overwriting information 
			FileWriter fw = new FileWriter(resolvePath(filename),false); 
			fw.write(data + "\r\n");
			fw.close();

		} catch(Exception e){}
	}

    
	// APPEND IN THE TEXT FILE SOME NEW VALUES
	private void append_txt(String filename, double data) {

		try {
			// FileWriter(filepath, boolean).
			// boolean = true -> appending information
			// boolean = false -> overwriting information 
			FileWriter fw = new FileWriter(resolvePath(filename),true); 
			fw.write(data + "\r\n");
			fw.close();

		} catch(Exception e){}
	}


	// APPEND IN THE TEXT FILE SOME NEW VALUES OF A VECTOR
	/*
	private static void writevector_txt(String filename, Vector3 data) {

		try {

		FileWriter fw = new FileWriter(filename,true); 
		fw.write(data.get(2) + ";" + data.get(3) + data.get(4) + "\r\n");
		fw.close();

		} catch(Exception e){}
	}
	*/

//	private void disableupdatescene() {
//
//		Simulation simulation_0 = getActiveSimulation();
//    		Scene scene_0 = simulation_0.getSceneManager().getScene("vorticity");
//    		SceneUpdate sceneUpdate_0 = scene_0.getSceneUpdate();
//    		sceneUpdate_0.setEnabled(false);
//    		sceneUpdate_0.setSaveAnimation(false);

//  	}

}
