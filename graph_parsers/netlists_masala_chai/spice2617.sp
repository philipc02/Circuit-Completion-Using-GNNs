spice
* Define power supply
VDD 4 0 DC VDD

* Define voltage sources for inputs
Vin 0 6 DC Vin
Vcont1 8 3 DC Vcont1
Vcont2 3 7 DC Vcont2

* Define NMOS Transistors
M3 2 2 3 3 NMOS
M4 2 7 3 3 NMOS
M5 3 3 8 8 NMOS
M6 3 3 8 8 NMOS

* Define PMOS Transistors
M1 2 0 3 3 PMOS
M2 2 6 3 3 PMOS

* Define resistors
RD 5 2 RD
RD 4 2 RD

* Define current source
Iss 3 8 DC Iss

* Define output
Vout 2 0 DC Vout

* End of netlist
.end