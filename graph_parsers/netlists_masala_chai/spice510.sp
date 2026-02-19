plaintext
* SPICE Netlist for the given Circuit

V1 1 0 DC Vs1
V2 3 0 DC Vs2
V3 5 0 DC Vsd

R1 2 1 R1
R2 4 3 R1

R3 5 6 R3
R4 6 0 R3
R5 6 7 R3

* Operational Amplifier connections
* Using node numbers:
* +IN is connected to node 2
* -IN is connected to node 4
* OUT is connected to node 5

XOPAMP 2 4 5 OPAMP

* .MODEL OPAMP OPAMP
* Note: OPAMP in SPICE often requires specific parameters which will not be covered here, since this is a generic template.

.END