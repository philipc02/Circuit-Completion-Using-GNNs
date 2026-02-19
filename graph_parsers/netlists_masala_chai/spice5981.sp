spice
* SPICE netlist for the given schematic

* Transistors
Q1 4 v1 3 NPN
Q2 5 4 VBIAS NPN

* Current Sources
I1 4 4 DC 2I
I2 2 2 DC I

* Nodes
* 1: Vi
* 2: GND
* 3: Vo
* 4: Intermediate node
* 5: Collector node of Q2

* Voltage Supply (if necessary, specify later)
*V1 VBIAS 0 DC <voltage_value>