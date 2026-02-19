* SPICE Netlist for the given circuit

* Voltage Sources
V1 8 0 DC 0
V2 2 7 DC 2.5
V3 6 7 DC 10

* Resistors
R1 8 1 15k
R2 1 2 10k
R3 3 4 5k

* Diodes
D1 1 2 Dmodel
D2 1 3 Dmodel

* Model for Diodes
.model Dmodel D

* Simulation .control commands
.control
run
.endc

.end