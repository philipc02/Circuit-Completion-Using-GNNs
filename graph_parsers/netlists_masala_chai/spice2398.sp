spice
* SPICE Netlist

* Voltage Sources
VDD 9 0 DC 1
VSS 5 0 DC 0

* Current Source
I1 1 2 DC 2.5uA

* PMOS Transistors
M1 7 2 2 2 PMOS W=5u L=0.5u
M2 9 2 3 3 PMOS W=5u L=0.5u
M3 9 2 3 3 PMOS W=5u L=0.5u

* NMOS Transistors
M4 2 4 5 5 NMOS W=0.5u L=4u
M5 2 2 5 5 NMOS W=5u L=0.5u
M6 8 3 5 5 NMOS W=30u L=0.4u
M7 8 3 5 5 NMOS W=30u L=0.4u

.end