spice
* Netlist for given schematic

* Voltage source V_n,in
Vn_in 3 4 DC 0

* Capacitor C_in
Cin 3 4 1uF

* NMOS Transistor M1
M1 2 3 4 4 NMOS

* Resistor R_D
RD 5 2 1k

* Voltage source V_DD
VDD 6 5 DC 5V

* NMOS model
.model NMOS NMOS (level=1)

* End of netlist
.end