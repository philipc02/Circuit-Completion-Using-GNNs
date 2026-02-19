spice
* Netlist for given circuit

* Current Source
IIN 4 5 DC <current_value> * specify current value

* Transistors
Q1 7 3 8 NPN
Q2 7 4 8 NPN
Q3 2 1 8 NPN
Q4 4 2 8 NPN

* Voltage Source
VIN 1 8 DC <voltage_value> * specify input voltage value
VCC 5 8 DC <vcc_value> * specify VCC voltage value

* Analysis and other commands can be added here
.end