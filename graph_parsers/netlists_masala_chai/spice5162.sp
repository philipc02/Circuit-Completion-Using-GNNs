spice
* SPICE Netlist for given schematic
* Components
C1 7 5 1n
C2 5 2 1n
R 2 5 30k
R1 5 3 10k
R2 2 5 15k

* Ideal op-amp (use symbol "E" for Voltage Controlled Voltage Source)
* This models an ideal op-amp with very high gain
* Node 2 is output, node 5 is inverting input, node 3 is non-inverting input (ground)
EOPAMP 2 3 5 3 1MEG

* Input and Output
Vin 7 0 DC 0V
Vout 2 0

* End of netlist