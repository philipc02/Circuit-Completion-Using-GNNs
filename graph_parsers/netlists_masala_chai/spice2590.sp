spice
* NMOS and PMOS model definitions
.model PMOS PMOS (LEVEL=1)

* Transistors
M1 4 3 1 1 PMOS
M2 2 3 1 1 PMOS

* Current Source
Iss 3 6 DC 1m

* Voltage Sources
V1 3 5 DC (Vin1-Vin2)/2
V2 2 1 DC (Vin2-Vin1)/2

* End of netlist