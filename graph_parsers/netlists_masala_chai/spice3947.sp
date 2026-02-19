plaintext
* SPICE Netlist for the given circuit

Vs 4 3 DC 0
Vcc 7 3 DC 2.5
Vee 8 3 DC -2.5

R1 7 9 40k
R2 9 8 60k
RE 8 5 2k
RC 6 3 4k

Cc 9 4 0.1u ; Assuming value for coupling capacitor

Q1 6 9 5 NPN

* .model line can be added if needed, e.g. .model NPN NPN(...)