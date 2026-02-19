spice
* SPICE netlist for the given schematic

* Voltage Source
V1 2 0 DC 5

* Current Source
I1 2 0 DC 200u

* PNP Transistors
Q1 4 3 2 2 2N3906
Q2 7 3 2 2 2N3906

* NPN Transistors
Q3 4 3 5 5 2N3904
Q4 7 3 6 6 2N3904

* Resistors
R1 4 5 1k
R2 7 6 1k
Ry 3 5 50k
Rx 3 5 50k

* Additional Voltage Source for V6 (if needed)
V6 6 0 DC 5

* Model Definitions 
.model 2N3906 PNP
.model 2N3904 NPN

.end