spice
* Netlist for the given schematic

* Voltage Sources
VDD 3 5 DC <VDD_value>
Vn1 7 2 DC <Vn1_value>
Vn2 2 5 DC <Vn2_value>
VnRD1 3 5 DC <Vn,RD1_value>
VnRD2 5 5 DC <Vn,RD2_value>

* Current Source
Iss 6 0 DC <Iss_value>

* Resistors
RD1 3 3 <RD1_value>
RD2 5 5 <RD2_value>

* NMOS Transistors
M1 3 7 6 6 NMOS
M2 3 2 6 6 NMOS

* Specify the model parameters for NMOS
.model NMOS NMOS (level=1 Vto=0.7 Kp=20u)

* Additional info
* Net 0 is assumed to be ground
* Replace placeholder values like <VDD_value>, <Vn1_value> with actual values