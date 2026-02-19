plaintext
* SPICE Netlist for the given schematic
*
* Components
M1 2 3 2 2 NMOS
R1 6 2 1k
R2 2 7 1k
RD 4 2 1k
RS 2 3 0.25k

* Voltage Sources
V1 6 0 DC 10
V2 7 0 DC -10
V3 4 0 DC 5
V4 3 0 DC -5

* Model Definitions
.model NMOS NMOS (Level=1)

* End of Netlist