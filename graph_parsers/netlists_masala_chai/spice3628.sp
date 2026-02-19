spice
* SPICE netlist for the given circuit

* Diodes
D1 1 2 D_model
D2 3 4 D_model

* Resistors
R1 4 v0 1000
R2 2 5 1000

* Voltage Source at v0
V0 v0 0 DC 5

* Model Definitions
.model D_model D