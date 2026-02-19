spice
* SPICE Netlist

VIN 1 0 DC 0V
V1 4 0 DC 5V

I1 2 8 DC 0.1m
I2 3 9 DC 0.1m

R1 2 3 500
R2 4 5 25k

Q1 2 2 8 NPN
Q2 5 3 9 NPN

* Connections based on node numbers:
* Q1: B=2, C=2, E=8 [Left Transistor, NPN]
* Q2: B=3, C=5, E=9 [Right Transistor, NPN]

.END