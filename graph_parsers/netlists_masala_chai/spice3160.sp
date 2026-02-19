* Resistors
R1 1 2 RF
R2 5 6 RF

* Capacitors
C1 1 3 C1
C2 6 5 C1
C3 2 3 C2
C4 6 5 C2
C5 2 4 1pF
C6 4 5 1pF

* Voltage Sources (DC Analysis)
V1 3 0 DC Vin

* Operational Amplifier
* Connections: Non-inverting input (6), Inverting input (8), Output (5)
XOP1 6 8 5 OPAMP

* End of Netlist
.end