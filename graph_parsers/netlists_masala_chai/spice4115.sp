plaintext
* SPICE Netlist for the Circuit

* Voltage Sources
V1 3 0 DC 12V
V2 7 0 DC

* Resistors
RB 3 4 1MEG
RC 3 6 5.1K
RS 7 6 1K
RL 4 2 500K

* Capacitors
CC 6 0 10U
CL 2 0 10P

* NPN Transistor
Q1 5 4 6 NPN

* Simulation Commands
.TRAN 1u 10m
.END