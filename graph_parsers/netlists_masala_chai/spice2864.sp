* NMOS Transistors
M1 3 1 0 0 NMOS
M2 2 5 3 3 NMOS

* Resistor
RD 4 2 RD_VALUE

* Voltage Source
VDD 4 0 VDD_VALUE

* Nodes
* 1 = Vin
* 2 = Vout
* 3 = X (connection node between M1 and M2)
* 4 = VDD
* 5 = Vb

.model NMOS NMOS_MODEL_PARAMETERS