plaintext
* SPICE netlist for the given analog circuit
* Voltage Source
Vsig 9 0 DC 0

* Resistors
R_sig 8 9 R_sig
R_L 10 6 R_L
r_o 3 7 r_o

* Capacitors
C_gd 5 7 C_gd
C_gs 5 9 C_gs
C_L 3 11 C_L

* Voltage-Controlled Current Sources
Gm1 7 2 5 6 gm
Gmb 3 2 10 6 gmb

* Connections
* Note: Nets are given by numbers from the annotated image
.END