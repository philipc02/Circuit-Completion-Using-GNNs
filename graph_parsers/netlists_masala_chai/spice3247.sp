plaintext
* MOSFET Definitions
M1 5 6 3 3 NMOS
M2 2 6 3 3 NMOS
M3 5 8 5 5 PMOS
M4 2 9 5 5 PMOS

* Current Source
I1 3 0 DC 1mA

* Voltage Source
VDD 8 0 DC 5V
VIN 6 0 DC 1V

* Analysis
.tran 1ns 10ns
.end