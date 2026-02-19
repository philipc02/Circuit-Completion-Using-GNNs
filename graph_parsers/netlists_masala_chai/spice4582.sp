spice
* SPICE Netlist

* Voltage Sources
V1 2 0 DC 5V

* PMOS Transistors
M_P1 vO1 2 2 2 PMOS
M_P2 vO2 2 3 3 PMOS

* NMOS Transistors
M_NA vO1 vA 4 4 NMOS
M_NB vO1 vB 4 4 NMOS
M_NC vO1 vC 6 4 NMOS
M_N1 4 CLK 8 0 NMOS
M_N2 vO2 vO1 5 5 NMOS

.END