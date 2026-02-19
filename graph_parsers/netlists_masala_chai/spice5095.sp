spice
* SPICE Netlist for the given schematic
* Connections are based on the annotated image.

* Voltage source
V1 6 0 AC 1

* Resistors
R1 6 2 1k
R2 2 4 1k
R3 4 5 100k

* Operational Amplifier
* Non-inverting input connected to node 3, inverting input to node 2, output to node 5
XOPAMP 3 2 5 LF157A

* Ground connection
V+ 3 0 DC 0
V- 3 0 DC 0

.end