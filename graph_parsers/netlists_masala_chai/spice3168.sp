spice
* SPICE Netlist for the provided schematic

* Voltage Source
VDD 6 0 DC 5V

* Current Source
Iout 3 0 DC 1A

* Resistors
R1 X 0 1k
R3 2 4 1k
rO4 2 3 1k
rO2 4 5 1k
RS 5 0 1k

* MOSFETs
M4 2 3 6 6 PMOS
M2 4 3 5 5 NMOS

* Control statements
.include 'mosfet.lib'
.TRAN 1n 100n
.END