spice
* SPICE Netlist

* Voltage Source
V1 7 0 DC

* Resistors
RS 7 2 R_S
RL 4 6 R_L
R2 3 8 R_2

* NPN BJTs
Q1 5 2 0 QNPN
Q2 4 3 0 QNPN

* Model Definitions
.model QNPN NPN (parameters...)

* Nodes
* 1: Ground
* 2: Connection between RS, Q1 base
* 3: Connection between Q2 base, R2
* 4: Vo at RL
* 5: Q1 collector
* 6: RL to ground
* 7: Vi (input voltage)
* 8: Connection between R2, Q1 collector and Q2 collector

.end