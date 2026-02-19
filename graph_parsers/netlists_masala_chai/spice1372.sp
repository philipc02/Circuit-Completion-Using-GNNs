spice
* Netlist for the given circuit
* Transistors
M1 2 0 3 3 NMOS
M2 2 2 3 3 PMOS

* Resistors
Rro1 3 GND ro1
Rro2 2 out ro2

* Current source
Igm1 0 3 gm1*Vin

* Voltage source
Vin in 0 DC Vin

* Connections
*out is the output node.
* Vin is the input voltage source.