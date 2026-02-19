spice
* SPICE Netlist
V1 4 0 DC 0

* Voltage Source
Vvi 4 0 DC Vi

* MOSFET
M1 2 3 6 6 NMOS

* Current Source
Iq 5 2 DC Iq

* Capacitors
Cc1 3 4 Cc1
Cc2 2 1 Cc2

* Resistors
Rg 3 0 200k
Ro 2 1 Ro
Rl 1 0 Rl

* Voltage sources for supply
Vplus 5 0 DC 5
Vminus 6 0 DC -5

.end