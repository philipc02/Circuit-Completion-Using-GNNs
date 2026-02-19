spice
* NMOS Transistor Circuit
M1 1 2 3 3 NMOS

* Resistors
R1 5 2 1.5MEG
R2 2 0 1MEG
RD 1 7 10K
RS 3 6 22K

* Voltage Source at Node 5
V1 5 0 DC 0

* Ground at Node 0
V0 0 0 DC 0

* Specify Model for NMOS
.model NMOS NMOS (LEVEL=1)