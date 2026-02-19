spice
* Spice netlist for the given circuit

* Current Source
Is 3 5 DC <value_of_Is>

* Resistors
RS 2 5 <value_of_Rs>
RF 2 4 <value_of_Rf>

* Voltage Amplifier (Op-amp)
X1 3 2 4 OPAMP

* Ground
Vground 5 0 DC 0

.model OPAMP opamp

.end