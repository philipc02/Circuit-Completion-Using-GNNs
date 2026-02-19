spice
* SPICE Netlist for the given circuit

* Resistors
R_RS 8 E RS
R_RE B E RE
R_ro 3 4 ro
R_ROC 2 C Roc

* Voltage Source
V_X C 5 Vx

* Dependent Source
G_gm 4 B E V_pi gm

* Voltage-controlled voltage source to reflect V_pi
E_Vpi B E VALUE = {V(1) - V(E)}

.tran 0.1s 10s
.end