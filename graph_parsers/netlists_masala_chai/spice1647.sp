spice
* SPICE Netlist for BJT Circuit

* Transistor
Q1 4 5 E QMOD

* Capacitors
C1 2 4 Cmu
C2 2 5 Cpi
C3 4 2 Ccs

.model QMOD NPN(IS=1E-14 BF=100)

* Node voltage definitions
VEE E 0 0