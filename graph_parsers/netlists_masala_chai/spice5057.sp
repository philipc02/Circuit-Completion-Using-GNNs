spice
* SPICE netlist for the provided schematic

V1 4 0 DC Vcc      * Voltage source Vcc
R1 3 5 R_value     * Resistor with value R_value
Q1 4 5 2 NPN       * NPN transistor
D1 5 0 D_model     * Diode model

* Connections:
* V1: Vcc connected to node 4
* R1: Resistor connected between nodes 3 and 5
* Q1: Base connected to node 5, collector connected to node 4, emitter connected to ground (node 2)
* D1: Diode anode connected to node 5, cathode to ground

.end