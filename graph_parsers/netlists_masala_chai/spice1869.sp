plaintext
* Netlist for given circuit

* Transistors
M1 N3 Vb N3 N3 NMOS
M2 N2 N4 N5 N5 PMOS

* Current Source
Iin N3 0 I_in

* Voltage Source
VDD N5 0 V_DD

* Resistors
RD1 N5 N6 R_D1
R1 N2 N7 R_1
R2 N7 0 R_2
RF N3 N7 R_F

* Nodes
N1 Vout
N2 N4

* End of netlist