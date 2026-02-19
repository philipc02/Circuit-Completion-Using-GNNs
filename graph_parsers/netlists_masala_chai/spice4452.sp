spice
* SPICE Netlist for the given Schematic

V1 4 1 DC <value_of_Vs>   * Voltage Source Vs

R_S 4 2 5k              * Resistor Rs
R_F 2 3 10k             * Resistor Rf
R_1 5 1 50k             * Resistor R1
R_2 3 1 10k             * Resistor R2
R_L 5 6 4k              * Resistor Rl

* Op-Amp (Generic)
XOP 3 2 5 opamp         * Op-Amp with inputs: inverting (3), non-inverting (2) and output (5)

.model opamp opamp      * Generic op-amp model

.end