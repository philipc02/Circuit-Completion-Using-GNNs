spice
* SPICE netlist for the circuit

* Voltage Source
VCC 2 1 DC 2.5V

* Current Source
IREF 2 4 DC IREF_value

* Resistor
RC 2 6 RC_value

* Transistors
QREF 4 4 1 PNP_MODEL
Q1 3 4 1 PNP_MODEL
Q2 6 3 1 NPN_MODEL

* Assigning Models
.model PNP_MODEL PNP
.model NPN_MODEL NPN

* End of netlist
.end