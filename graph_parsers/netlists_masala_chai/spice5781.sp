* Transistor Circuit

* Resistors
R1 4 2 6.2k
R2 6 2 110k
R3 2 3 10k

* Voltage Sources
V1 4 0 DC 3V
V2 6 0 DC 0.75V
V3 3 0 DC -3V

* NPN Transistor
Q1 2 6 3 NPN

.model NPN NPN (IS=1E-14 BF=100)

* End of Netlist