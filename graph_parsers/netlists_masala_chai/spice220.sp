spice
*SPICE netlist

*C1 and C2 are capacitors
C1 5 2 
C2 2 3 

*Operational Amplifier
*Opamp ideal model
* The negative terminal (-) of the op-amp is connected to ground
XOP 2 1 4 OPAMP

*Voltage output
VOUT 4 6 DC 0

*Ground
VSS 1 0

*End of netlist