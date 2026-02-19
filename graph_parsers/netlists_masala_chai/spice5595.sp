spice
* SPICE netlist for OpAmp Circuit

* Voltage Source
V1 7 8 DC 0

* Resistors
R1 1 3 10k
R2 2 3 10k
R3 1 2 10k
R4 8 2 10k
R5 4 5 20k
R6 2 5 30k

* Operational Amplifiers (ideal)
XOP1 3 2 4 OpAmp
XOP2 2 7 2 OpAmp

* Model Definitions
.model OpAmp opamp

* End of netlist