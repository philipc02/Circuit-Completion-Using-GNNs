spice
* CMOS Logic Circuit Netlist

* PMOS Transistors
M1 3 1 3 3 PMOS
M2 4 2 3 3 PMOS

* NMOS Transistors
M3 4 2 2 2 NMOS
M4 4 1 2 2 NMOS

* Voltage sources
V1 3 0 Vdd

* Inputs
VIN_A 1 0 DC
VIN_B 2 0 DC

* Output
* Y at node 4

* Models
.model PMOS PMOS
.model NMOS NMOS

.END