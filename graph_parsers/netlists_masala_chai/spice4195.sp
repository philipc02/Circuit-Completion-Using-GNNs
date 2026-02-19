* SPICE Netlist for the given Circuit

* Voltage Sources
V1 1 0 DC V1_DC
V2 3 0 DC V2_DC

* Resistors
R1 vin 1 1k
R2 1 2 10k
R3 2 0 1k
R4 2 3 1k
R5 3 5 20k

* Operational Amplifiers
* Assuming Ideal Opamp Model
XU1 0 1 2 opamp
XU2 0 3 vout opamp

* Model Definition for Ideal Opamp
.model opamp opampmodel