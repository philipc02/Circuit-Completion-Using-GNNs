plaintext
* SPICE netlist for the circuit

V1 9 11 DC Vdd/2

R1 9 5 Rnm
R2 5 4 Rnm
R3 4 6 2Rnp
R4 6 7 Rnp
R5 7 10 Rop

I1 5 11 DC InmVin/2
I2 7 2 DC IopVop

.END