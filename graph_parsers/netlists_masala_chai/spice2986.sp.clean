spice
* SPICE Netlist for the Given Schematic

M1 5 Vin 0 0 NMOS
M2 4 2 0 0 NMOS

RD1 5 3 3k
RD2 3 VDD 3k
R1 2 4 1k
R2 Vin 0 2k

CGS2 2 0 1pF

VDD VDD 0 DC 5V
Vin Vin 0 SIN(0 1V 1kHz)

* Models
.model NMOS nmos level=1

.end