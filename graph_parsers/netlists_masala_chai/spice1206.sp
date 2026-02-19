spice
* SPICE Netlist

* Voltage Source
VDD 3 5 DC VDD

* Resistor
RD 3 6 RD

* NMOS Transistor
M1 3 2 4 4 NMOS

* PMOS Transistor
M2 3 6 3 3 PMOS

* Current Source
I1 4 5 DC I1

* Nodes
* Node 2: Vin
* Node 3: VDD, Connected to RD, Drain of M1, Source of M2
* Node 4: Source of M1, Negative of I1
* Node 5: Ground
* Node 6: Vout, Vb

.END