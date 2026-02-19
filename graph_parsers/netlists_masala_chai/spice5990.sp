plaintext
* SPICE Netlist

V1 1 0 DC 5V
V2 8 0 DC -5V
I1 3 1 DC 0 ; Current source (left)
I2 9 2 DC 0 ; Current source (right)

Q1 3 4 5 NMOS
Q2 6 4 7 NMOS
Q3 1 3 2 PMOS
Q4 4 4 2 PMOS
Q5 6 4 9 PMOS
Q6 7 8 8 NMOS

R1 7 0 R

* Model definitions
.model NMOS NMOS
.model PMOS PMOS

.END