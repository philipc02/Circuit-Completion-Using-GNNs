* SPICE Netlist for the given circuit

V1 3 0 DC Vin
M1 2 1 3 3 NMOS
RD 2 VDD 1k
RS 1 3 1k
VDD VDD 0 DC VDD

* Connections
* 1 - Source of M1, connected to RS
* 2 - Drain of M1, connected to RD
* 3 - Gate of M1 connected to Vb and Source of Vin

.end