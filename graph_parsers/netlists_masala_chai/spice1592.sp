spice
* NMOS Transistor M1
M1 X 1 3 3 NMOS

* NMOS Transistor M2
M2 Y 2 3 3 NMOS

* Drain Resistors
RD1 5 4 RD
RD2 6 4 RD

* Current Source
I1 3 0 I_ss

* Source Resistor
RSS 3 0 R_SS

* Voltage inputs
Vin1 1 0 DC 0V
Vin2 2 0 DC 0V

* Supply Voltage
VDD 4 0 DC V_DD

* Connections
X Vin1 node at netlist 1
Y Vin2 node at netlist 2
P node at netlist 3