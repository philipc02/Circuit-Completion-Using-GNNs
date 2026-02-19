plaintext
* Example SPICE Netlist
* VDD = 2.5 V

VDD 9 0 DC 2.5V

* Transistors
Q1 8 10 7 NPN
Q2 5 9 7 NPN
Q3 4 1 7 NPN
Q4 2 9 7 NPN
Q5 3 2 7 NPN
Q6 2 6 7 NPN

* Resistors
R1 9 2 2k
R2 10 7 15k
R3 6 7 15k

* Voltage Nodes
V1 4 0 DC 0V
V2 8 0 DC 0V

.END