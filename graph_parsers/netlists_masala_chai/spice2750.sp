plaintext
* SPICE Netlist for the given schematic
VDD 32 0 DC 5
Vin Vin 0 DC 1

*MOSFETs
M1 X Vin P NMOS
M2 4 2 3 NMOS
M3 32 2 2 PMOS
M4 32 2 4 PMOS

*Current Source
I1 5 0 DC ISS

*Capacitor
C1 4 3 CL

* Nodes
* Node 32 - VDD
* Node 2  - X
* Node 4  - Vout
* Node 3  - V1 (ground reference)
* Node 5  - P

* Model definitions (example, replace with actual parameters)
.model NMOS NMOS
.model PMOS PMOS