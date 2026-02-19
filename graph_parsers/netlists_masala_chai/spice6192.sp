spice
* SPICE Netlist for the given schematic

* Voltage Source
Vs 7 4 DC <value> ; Assuming a DC voltage source, replace <value> with the actual voltage

* Resistors
Rs 7 4 <value> ; Replace <value> with resistance of Rs
R1 4 8 <value> ; Replace <value> with resistance of R1
R2 4 3 <value> ; Replace <value> with resistance of R2
Rout 3 6 <value> ; Replace <value> with resistance of Rout
RL 3 2 <value> ; Replace <value> with resistance of RL

* Operational Amplifiers (Assuming ideal)
* Op-amp A1
A1 4 4 4 opamp

* Op-amp A2
A2 4 4 3 opamp

* Op-amp A3
A3 3 4 3 opamp

* .MODEL statement for opamps (if needed)
.model opamp opamp

* Simulation commands
.control
run
.endc

.end