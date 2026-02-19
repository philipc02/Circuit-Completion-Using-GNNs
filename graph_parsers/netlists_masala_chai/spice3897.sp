plaintext
* SPICE Netlist for Given Circuit
VCC 5 0 DC 12V
VEE 4 0 DC -12V
V1 1 0 DC <value> ; Replace <value> with the required input voltage value

R1 1 3 15k
R2 3 4 100k
RC 5 2 2.2k

Q1 2 3 4 NPN

* Define model for NPN
.model NPN NPN