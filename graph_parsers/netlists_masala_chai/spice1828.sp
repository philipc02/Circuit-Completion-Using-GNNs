plaintext
* SPICE Netlist

* Voltage Source
Vin 1 3 DC <Vin_value>

* Current Source
I1 2 0 DC <I1_value>

* NMOS Transistor
M1 2 1 3 3 NMOS

* Control loop (Voltage controlled voltage source)
E1 2 3 2 3 K

* Load
Vout 2 0 DC 0

* Model Definitions
.model NMOS NMOS

.end