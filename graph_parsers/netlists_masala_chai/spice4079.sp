spice
* SPICE Netlist

* Voltage Source
V1 7 8 DC Vi

* Resistors
RS 4 3 RS
RB1 8 3 RB1
RO1 5 3 ro1
RO2 6 3 r_o2

* Capacitors
CX1 3 3 Cx1
CM1 3 5 CM1
CMU1 5 2 Cmu1
CPI1 8 3 Cpi1
CN2 2 3 Cn2

* Dependent Current Source
G1 5 3 VPI1 gm1

* Node assignments:
* 1. Vi positive terminal: 7
* 2. Vi negative terminal: 8
* 3. General node: 3
* 4. RS connection node: 4
* 5. Current source input node: 5
* 6. RO2 input node: 6
* 7. Ground node: Com