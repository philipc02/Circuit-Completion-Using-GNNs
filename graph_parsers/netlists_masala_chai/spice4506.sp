* SPICE Netlist

* Voltage Input
Vi 1 0 DC 0

* Capacitors
C1 1 2 C
C2 2 3 C
C3 3 4 C

* Resistors
R1 3 5 R/3.546
R2 3 0 R/1.392
R3 4 0 R/0.2024

* Op-Amp
* Positive input: node 4, Negative input: node 2, Output: node 2
XOPAMP 4 2 2 OPAMP

* .OPAMP model
.model OPAMP