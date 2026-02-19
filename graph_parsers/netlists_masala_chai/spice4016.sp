spice
* SPICE netlist for the given schematic

* Node declarations
V1 Vi 5 DC 0
GND 0

* Current source
IQ 5 7 DC 2mA

* MOSFET: Assuming NMOS with nodes: drain (2), gate (5), source and body (7)
M1 2 5 7 7 NMOS

* Resistors
RC 2 6 1.5k
RL 6 2 2.5k

* Capacitor
C1 6 2 <capacitance_value> 

* DC voltage source
Vi 1 5 DC <input_voltage_value>

.END