plaintext
* SPICE Netlist for the given schematic

* Voltage Sources
VCC 5 0 DC <value_of_VCC>
VEE 0 2 DC <value_of_VEE>

* MOSFET
M1 5 2 4 4 NMOS

* Resistors
RLoad 5 5 <value_of_LOAD>
R1 2 0 <value_of_R>

* Op-Amp
* Assuming ideal Op-Amp model reference for simulation
* XU1 0 2 5 OpAmpIdeal

.model NMOS NMOS (LEVEL=1)

.end