spice
* SPICE Netlist for the given circuit
M1 2 4 3 3 NMOS
M2 2 3 0 0 NMOS
R1 2 3 R1_VALUE
RD 2 0 RD_VALUE
VBD 4 0 Vbd_VALUE
VIN 4 0 Vin_VALUE

* Specify additional model statements or parameters here
.model NMOS nmos level=1