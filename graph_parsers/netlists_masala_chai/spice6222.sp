plaintext
* SPICE Netlist

Q1 3 5 2 npn
Q2 2 3 4 npn
Q3 3 2 4 npn
Q4 2 4 4 npn
Q5 6 1 6 npn

I1 6 5 DC IBIAS

R1 1 6 R1
R2 6 0 R2
RL 2 7 RL

VCC 5 0 DC VCC
VEE 4 0 DC VEE
VIN 1 0 AC 1

.control
tran 1n 100n
.endc
.end