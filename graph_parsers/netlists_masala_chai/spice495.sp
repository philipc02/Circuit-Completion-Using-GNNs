spice
* SPICE Netlist

R13 6 2 33k
I13 4 3 DC 2 ; Current source i_{13}
V13 2 0 DC 2 ; Voltage source v_{13}

* Assuming Q13 is NMOS
M_Q13 7 2 2 2 NMOS

* Model definitions
.model NMOS NMOS