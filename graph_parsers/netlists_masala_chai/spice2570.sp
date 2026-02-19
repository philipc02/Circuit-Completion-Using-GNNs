spice
* SPICE Netlist for the given schematic

VDD 22 0 DC VDD

RD1 22 3 RD
RD2 7 4 RD

M1 3 6 8 8 NMOS
M2 4 5 8 8 NMOS

ISS 8 0 DC ISS

* Inputs
Vin1 6 0 DC 0
Vin2 5 0 DC 0

* Outputs
Vout1 3 0
Vout2 4 0

.END