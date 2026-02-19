spice
* SPICE Netlist
* Nodes: 1, 2, 3, 4, 5, 6

* Input Current Source (symbol ic)
Iin 2 1 Ii

* Resistor Ri
R1 4 6 Ri

* Voltage-Controlled Current Source (symbol gen2)
G1 5 6 VALUE = {A * V(4, 6)}

* Resistor Ro
R2 3 5 Ro

* Output Current
Iout 2 3 Io

* Ground
Vgnd 6 0 0