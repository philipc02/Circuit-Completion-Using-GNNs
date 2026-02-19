spice
* Transistor components
M1 7 6 2 2 PMOS
M2 4 5 2 2 PMOS
M3 4 7 3 3 NMOS
M4 3 4 4 3 NMOS

* Current source
I1 2 0 DC VDD

* Voltage source
V1 3 0 DC -VSS

* Additional notes
* Nodes 6 and 7 are the gate voltages for M1 and M3, respectively.
* Nodes 5 is the gate voltage for M2.
* Node 4 is a shared node connecting M2, M3, and M4.
* Node 3 reference for NMOS transistors.