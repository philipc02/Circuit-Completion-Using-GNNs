spice
* SPICE Netlist for the given schematic

* Voltage Source
Vsig 5 0 DC 0

* Current Source
I1 4 6 DC 0.5mA

* Resistors
R1 5 8 50
R2 4 0 100k
R3 1 2 5k

* Capacitors
C1 4 0 inf
C2 4 1 inf

* Transistor
Q1 1 4 0 NPN

* Connections
* Node 5 is connected to ground
* Node 4 connects R2, I1, C1, base of Q1
* Node 6 connects the top of I1
* Node 1 connects collector of Q1 and R3
* Node 2 connects the other terminal of R3 and Vout

* Analysis
.tran 1n 10u
.end