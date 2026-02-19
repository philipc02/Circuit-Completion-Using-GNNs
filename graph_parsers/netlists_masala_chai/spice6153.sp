plaintext
* SPICE Netlist
V1 2 0 DC 20V
V2 2 1 DC 5V
Vsig 5 0 AC 1V
R1 2 2 10k
R2 1 5 10k
I1 3 0 DC 0.5mA

* Transistors
Q1 2 1 3 NPN
Q2 2 2 1 NPN

* Analysis
.OP
.AC DEC 10 1 100k
.END