plaintext
* NMOS M1: Source connected to RS, drain to M2 and gate to Vin
M1 6 2 6 NMOS

* PMOS M2: Source connected to VDD, drain to output, and gate to M1
M2 7 8 6 PMOS

* Resistor RD: Connected between VDD and drain of M2
RD 3 7 1000

* Resistor RS: Connected between source of M1 and ground
RS 6 0 1000

* Voltage source VDD
VDD 3 0 DC 5V

* Input voltage source Vin
Vin 2 0 DC 1V

* Output
Vout 5