spice
* SPICE Netlist for Given Schematic

V1 4 6 DC <value_of_Vs> ; Voltage source Vs between nodes 4 and 6

RS 6 4 <value_of_RS> ; Resistor Rs between nodes 6 and 4

RE 4 2 <value_of_RE> ; Resistor Re between nodes 4 and 2

RPi 8 2 <value_of_Rpi> ; Resistor Rpi between nodes 8 and 2

Gm 5 7 2 8 <value_of_gm> ; Voltage-controlled current source gmVpi from nodes 5 to 7 controlled by voltage across 2 and 8

RC 5 7 <value_of_RC> ; Resistor Rc between nodes 5 and 7

RL 7 9 <value_of_RL> ; Resistor RL between nodes 7 and 9

* The following is assumed ground
R1 8 0 1e12 ; Large resistor connected to ground due to node not explicitly containing ground symbol

.end