spice
* SPICE Netlist for the given circuit

V1 5 0 DC 5V
V2 6 0 DC -5V
IREF 3 6 DC VALUE

R1 3 1 R1_VALUE
RC2 4 0 RC2_VALUE

Q1 5 3 1 NPN
Q2 5 2 4 NPN

.END