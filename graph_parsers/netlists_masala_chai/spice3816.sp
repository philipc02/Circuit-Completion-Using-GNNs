spice
* Netlist for CMOS Inverter

VDD 3 0 DC 10V
Vi  4 0 DC

* Transistors
* PMOS: ML (Drain 3, Gate 2, Source 3, Bulk connected to Source)
M1 3 2 3 3 PMOS

* NMOS: MD (Drain 2, Gate 4, Source 5, Bulk connected to Source)
M2 2 4 5 5 NMOS

* Ground Connection
V5 5 0 0V