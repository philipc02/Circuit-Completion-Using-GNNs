spice
* SPICE Netlist
* NMOS Transistors
M1 8 5 9 9 NMOS
M2 33 3 9 9 NMOS

* PMOS Transistors
M3 6 4 2 2 PMOS
M4 6 7 2 2 PMOS

* Current Sources
I1 8 0 DC BIAS2
I2 33 0 DC BIAS2
I3 9 0 DC BIAS1

* Voltage Source
V1 6 0 DC VBIAS

* Net Definitions
* 0: Ground
* 2: OUT+
* 3: IN-
* 4: OUT-
* 5: IN+
* 6: VBIAS
* 7: Internal node
* 8: BIAS2
* 9: BIAS1
* 33: BIAS2