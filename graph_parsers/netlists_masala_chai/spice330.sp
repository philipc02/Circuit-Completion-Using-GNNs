spice
* SPICE Netlist for the Circuit

V1 8 0 DC 0           * Voltage source vi

RS 6 5 1k             * Resistor RS, assuming a resistance value of 1k Ohms
RL 3 33 1k            * Resistor RL, assuming a resistance value of 1k Ohms

M1 2 6 0 0 NMOS L=1u W=1u   * NMOS M1 with drain at 2, gate at 6, source at 0
M2 7 5 5 5 PMOS L=1u W=1u   * PMOS M2 with drain at 7, gate at 5, source at 5

.end