* Components
* NMOS Transistor
M1 4 2 0 0 NMOS

* Capacitors
CGD1 2 4  CGD1_value
CL 3 0  CL_value

* Resistors
R01R03 4 3  R_value

* Voltage Source
Vin1 1 0  Vin1_value

* Nodes
* 1 - Vin1
* 2 - Node between Vin1 and CGD1
* 3 - Node Vout1
* 4 - Node connection for M1, CGD1, and R01||R03
* 0 - Ground

.model NMOS NMOS_level