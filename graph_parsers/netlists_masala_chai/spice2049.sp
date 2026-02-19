plaintext
* MOSFET Circuit Netlist
M1 3 2 3 3 NMOS
M2 2 4 4 4 PMOS
M3 5 2 3 3 NMOS
M4 2 2 2 2 PMOS

* Voltage Sources
VDD 4 0 DC 5V
Vin 2 0 DC 1V

* Analysis
.TRAN 1n 10n
.END