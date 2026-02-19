spice
* SPICE Netlist for the given schematic

Rpi 4 2 rpi_value
Ro 5 3 ro_value
G1 2 5 VALUE = {beta * I(Veb)}
Veb 4 2 DC Veb_value
Vec 3 7 DC Vec_value

* Define controlling current for the current source
I2 2 8 {Ib}

* Assuming some ground reference at node 7
* DC Voltage sources Veb and Vec provide biasing

.end