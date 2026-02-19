plaintext
* SPICE Netlist
* Voltages
Vdd 6 0 DC 3V
Vss 4 0 DC -3V

* Current source
I1 5 6 DC Iq

* MOSFETs
M1 2 7 4 4 NMOS
M2 2 5 4 4 NMOS
M3 2 2 4 4 NMOS
M4 8 2 4 4 NMOS

* Resistors
RD1 2 4 RD1
RD3 6 2 RD3
RS 9 10 RS

* Nodes
* 1 → \( v_1 \)
* 2 → \( v_{o1} \)
* 3 → \( v_2 \)
* 4 → Ground
* 5 → Connection of \( I_Q \) and M2
* 6 → \( V^+ \)
* 7 → \( v_1 \) input of \( M_1 \)
* 8 → Output \( v_o \)
* 9 → Connection to output resistor \( R_S \)
* 10 → \( v_o \)

.model NMOS NMOS
.end