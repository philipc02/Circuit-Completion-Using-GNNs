spice
* NMOS Transistor M1
M1 4 5 5 NMOS_model

* NMOS Transistor M2
M2 3 2 5 NMOS_model

* Current Source I_IN
IIN 4 5 DC VALUE

* Voltage source definitions
VDD 4 0 DC VALUE
VIN 5 0 DC VALUE

* Output node definition
VOUT 6 0

* Connect output to node 3
RLOAD 6 3 0