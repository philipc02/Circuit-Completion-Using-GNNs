plaintext
* SPICE Netlist for the given circuit

VCC 1 0 DC <VCC_value>

R1 1 2 <R1_value>
R2 2 7 <R2_value>
RE 5 3 <RE_value>

Q1 2 3 7 NPN
Q2 2 2 7 NPN
Q3 6 3 5 NPN

IO 6 3 DC <IO_value>

.model NPN NPN