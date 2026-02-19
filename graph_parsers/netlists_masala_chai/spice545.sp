plaintext
* SPICE Netlist for the provided schematic

* Parameters
.param VDD=15V
.param VSS=-15V

* Voltage Sources
VDD 1 0 DC VDD
VSS 12 0 DC VSS
VIN1 8 2 DC 0
VIN2 9 2 DC 0

* Current Sources
I1 1 2 DC 0.1m
I2 7 12 DC 0.1m
I3 6 11 DC 0.1m

* Transistors (Using generic model names)
* PMOS: Drain, Gate, Source
M1 2 8 9 PCH
M1X 5 9 9 PCH
* NMOS: Drain, Gate, Source
M2 2 8 7 NCH
M2X 5 9 7 NCH
M9 11 2 12 NCH
M6 6 5 7 NCH

* Capacitors
CC1 2 11 1p
CC2 5 6 1p

* Resistors
RZ1 2 11 10k
RZ2 5 6 10k

* Output
VOUT 4 5 DC 0

* Model Definitions
.model NCH NMOS
.model PCH PMOS

.end