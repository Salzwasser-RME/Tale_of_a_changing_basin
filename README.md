# Tale_of_a_changing_basin
In this repository you will find the script to the model that is discussed in the paper with the corresponding name by Ebner&amp;Bulian et al.

FWB_Med_SimonEtAl2017.py
Requires input (data from Laskar at al., 2004 ( http://vo.imcce.fr/insola/earth/online/earth/online/index.php)) and can be run independently.
It produces the freshwater budget of the Medterranean as introduced by Simon et al., 2017. This was de basis for the fwb input for event72_change-over-time.py and event72-comparison.py.

event72-comparison.py
requires: matplotlib.pyplot, matplotlib.ticker,  numpy  
Needs event72_functions.py and and two folders (DATA and FIGURES) in the same directory to run.
It calculates the behaviour of the 3 boxes for a given degree of restriction for a sinusoidal fwb, as well as for a constant one.

event72-change-over-time.py
requires: matplotlib.pyplot, matplotlib.ticker,  numpy  
Needs event72_functions.py and and one folders (FIGURES) in the same directory to run.
Calculates the response of the boxes to a changing restriction for a sinusoidal freshwater budget that can be set to change.
