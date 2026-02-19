plaintext
* SPICE Netlist
* PMOS: M1, M3
* NMOS: M2, M4, M5, M6
* Voltage Sources: VDD, Vi, VSS
* Resistor: Rs

VDD 6 0 DC <value>       ; Define the DC value
Vi 8 0 DC <value>
VSS 7 0 DC <value>

M1 5 2 4 4 PMOS L=<value> W=<value>
M2 4 3 7 7 NMOS L=<value> W=<value>
M3 2 6 6 6 PMOS L=<value> W=<value>
M4 3 2 4 4 NMOS L=<value> W=<value>
M5 3 2 4 4 NMOS L=<value> W=<value>
M6 4 8 7 7 NMOS L=<value> W=<value>

RL 5 0 <value>

* End of netlist