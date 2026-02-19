spice
* SPICE netlist for the given schematic
V1 4 0 DC
RC 5 3  ; assume RC is connected to V+ = 5V and net 3
RB 6 4 50k
RS 3 2 100
RE 2 1 ; assume RE is connected to net 2 and V- = -5V
CC 4 6 
CE 2 0 
Q1 3 6 2 NPN 

* Voltage sources
V+ 5 0 DC 5
V- 1 0 DC -5

* End of Netlist