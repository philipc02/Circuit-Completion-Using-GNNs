plaintext
* SPICE Netlist

M1 1 2 2 2 NCH
M2 6 5 2 2 NCH
M3 2 2 4 4 PCH
M4 2 5 4 4 PCH
M5 3 3 2 2 PCH
I1 3 2 DC <Current_Value>

* Voltage sources and potentials
VCC 3 0 DC <VCC_Value>
VB 2 0 DC <VBIAS_Value>

* Output
Vo 6 2

* End of netlist