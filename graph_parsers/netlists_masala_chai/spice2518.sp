M1 5 2 0 0 NMOS  ; M1 connected to V_in, ground
M2 9 6 5 5 NMOS  ; M2 connected between M1 and M3, gate connected to V_b1
M3 9 3 4 4 PMOS  ; M3 in cascode, source to M4, gate to V_b2
M4 7 4 9 9 PMOS  ; M4 at top, gate connected to V_b3

VDD 7 0 DC  ; VDD supply

* Node numbers:
* 1 = VDD
* 2 = V_in
* 3 = M3 gate
* 4 = M4 gate
* 5 = Ground
* 6 = M2 gate
* 7 = M4 drain
* 9 = V_out

* Types are assigned based on assumed technology:
.model NMOS NMOS
.model PMOS PMOS