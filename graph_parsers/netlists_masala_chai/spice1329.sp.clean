plaintext
* SPICE netlist for the given schematic

Vin 1 0 DC 0        * Voltage source
R1 1 4 1k           * Resistor, 1k ohm example value
C1 3 2 10uF         * Capacitor, 10uF example value
A0 3 4 0 OPAMP      * Op-Amp model call example

* Connections:
* Vin is connected to net 1; the other terminal is grounded (net 0)
* R1 is between nets 1 and 4
* The inverting input of A0 is node 3, connected to the output of R1 (node 4)
* C1 is between the output of the op-amp (node 3) and inverting input (node 2)
* The output of the op-amp is node 3
* OPAMP represents a generic op-amp model, assuming nodes like SPICE `X` subcircuits

* .MODEL statements and OPAMP subcircuit should be defined elsewhere in the SPICE deck.
.end