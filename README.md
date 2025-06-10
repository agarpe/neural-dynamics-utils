The scripts in this repository are mainly intended for the analysis of neuronal dynamics in intracellular recordings. These codes have been developed during the process of the thesis in Computer Engineering and Telecommunications, in the Computational Neuroscience and Natural Computing program. 
  
  `charact_utils.py` contains a library used in the rest of the codes. 
  
  `lasers/superpos_functions.py` contains a library mainly used for action potential waveform analysis.
  
  `invariants/invariants_functions.py` contains a library for the analysis of time intervals and their variability. 

A detailed documentation and a cleaner version of the codes is coming soon. 

If you use this code, please cite the corresponding articles:

> Garrido-Peña, A., Elices, I., & Varona, P. (2021). Characterization of interval variability in the sequential activity of a central pattern generator model. Neurocomputing, 461, 667-678. https://doi.org/10.1016/j.neucom.2020.08.093

> Garrido-Peña, A., Sanchez-Martin, P., Reyes-Sanchez, M., Levi, R., Rodriguez, F. B., Castilla, J., Tornero, J., & Varona, P. (2024). Modulation of neuronal dynamics by sustained and activity-dependent continuous-wave near-infrared laser stimulation [Publisher: SPIE]. Neurophotonics, 11(2), 024308. https://doi.org/10.1117/1.NPh.11.2.024308


# How to use

Install conda environment to avoid dependencies error:

	conda env create -f neural-dynamics-utils.yaml

	conda activate neural-dynamics-utils

Ready to go!

## Waveform superposition and analysis
1. Create a ini file as the one in data-test/STG
2. Detect spikes/bursts.

	  ```python detection.py data-test/Exp1.h5 data-test/Exp1.ini```
  
 	 A "extended" dataframe with all information will be saved 
  'Trial', 'Type', 'Column_id', 'Column_name', 'Time', 'Signal', 'Bursts_Index', 'Bursts_Times', 'Peaks_Index', 'Peaks_Times', 'Waveforms', 'OPath'
3. Plot superposition and analysis
  
  		python laser/superpos_burst_waveform.py data-test/Exp1.ini data-test/Exp1_extended_data.pkl
 	 Note that triplets need to be defined in ini file:
 	 
	    [Superposition]
	    triplets = 1 2 3 | 3 4 5 | 5 6 7 | 7 8 9 | 9 10 11 | 11 12 13
	    column_id = 1
		# Forces computing metrics instead of loading from file if existing
		compute_metrics = y
	    

	Warning: First time this script is called it will analyze and generate waveform metrics dataframe, file-extension_metrics.pkl. Next time you run the script it will load the dataframe if found, unless you force it with config parameter "compute_metrics"

	The resulting metrics dataframe has the following structure:
	
	

4. Compare metrics with laser parameters by running plot_power_wavelength_temperature.py with *_metrics.pkl file

		 python laser/plot_power_wavelength_temperature.py data-test/Exp1.ini ../data/data-test/Exp1_metrics.pkl

	Note that you need to modify the config file including this section with the experiment log:
		
		[Power-wavelengths]
		laser_trials = 2 4 6 8 10 12
		# power density applied
		powers = 250 550 250 441 26 250
		wavelengths = 830 830 1020 1020 1450 1450
		temperatures = 18 19 22 27 27 60	 
 
	
