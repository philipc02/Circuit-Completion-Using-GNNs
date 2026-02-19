spice
* PMOS transistor M1
M1 Vout Vin VDD VDD PMOS

* NMOS transistor M2
M2 Vout Vb 0 0 NMOS

* Resistor Rs
Rs Vin Vout 1k ; Assuming 1k Ohms, change as needed.

* Voltage Source
VDD VDD 0 DC 5V ; Assuming 5V, change as needed.

* .model statements for transistors
.model PMOS PMOS (LEVEL=1 KP=30u VTO=-1)
.model NMOS NMOS (LEVEL=1 KP=50u VTO=1)

* End of netlist
.end