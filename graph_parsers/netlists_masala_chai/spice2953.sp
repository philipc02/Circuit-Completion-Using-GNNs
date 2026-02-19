plaintext
* Netlist for the given schematic

*MOSFETs
M1 4 6 6 PMOS
M2 5 3 3 NMOS

*Voltage Source
Vt 2 3 DC <value>

*Resistors
RD1 4 5 <value>
RD2 5 VDD <value>
R1 3 2 <value>
R2 6 3 <value>

*Voltage Rail
VDD 5 0 DC <VDD_value>

*.model PMOS PMOS ...
*.model NMOS NMOS ...

.END