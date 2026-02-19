spice
* SPICE Netlist
* Components
R3 vin 2 82k
R4 2 3 82k
C1 3 4 100p
R1 9 4 56k
C2 4 0 100p
R2 4 7 15k
* Op-Amp
* Connection: (+) 4, (-) 3, output 7
XU1 4 3 7 opamp
* Voltage input 
vin vin 0 DC 0V
* Ground
V0 0 0 DC 0V
.END