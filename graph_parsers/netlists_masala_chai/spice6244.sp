plaintext
* SPICE Netlist

* NMOS Transistor
M1 2 3 0 0 NMOS
M2 4 4 0 0 NMOS

* PMOS Transistor
M3 2 1 5 5 PMOS
M4 4 4 5 5 PMOS

* Current Source
IREF 5 4 DC VALUE

* Net Definitions
* 1: V_BIAS
* 2: V_in + V_OV
* 3: Ground
* 4: Node connecting GND
* 5: I_REF

.MODEL NMOS NMOS
.MODEL PMOS PMOS
.END