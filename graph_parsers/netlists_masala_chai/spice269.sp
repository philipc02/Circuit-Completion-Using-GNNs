spice
* SPICE Netlist for the given schematic

VCC 4 0 DC 10V

* Current sources
I_C11 6 4 DC 1mA
I_B3 5 0 DC 100uA
I_B4 2 0 DC 100uA
I_C3 0 3 DC 500uA
I_C4 0 4 DC 500uA

* NPN BJTs
Q3 6 5 0 QNPN
Q4 6 2 0 QNPN

.model QNPN NPN(IS=1e-14 BF=100)

.end