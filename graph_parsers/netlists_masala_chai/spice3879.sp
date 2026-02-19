plaintext
* SPICE netlist for the BJT amplifier circuit

Q1 3 1 2 NPN           * BJT Q1 with collector at node 3, base at node 1, emitter at node 2

RB 1 6 10k             * Base resistor RB with 10k ohm connected between node 1 and ground (node 6)
RC 4 3 4k              * Collector resistor RC with 4k ohm connected between node 4 (+8V) and node 3
RE 2 5 4k              * Emitter resistor RE with 4k ohm connected between node 2 and node 5 (-8V)

VCC 4 6 DC 8V          * Voltage source VCC connected between node 4 (+8V) and ground (node 6)
VEE 6 5 DC 8V          * Voltage source VEE connected between ground (node 6) and node 5 (-8V)

.END