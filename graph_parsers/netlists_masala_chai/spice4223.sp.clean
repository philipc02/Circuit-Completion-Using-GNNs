* SPICE netlist for the given schematic
* V+ and V- as power supply
* BJTs with specified nodes for collector, base, emitter (C B E)

IREF 3 8 DC 0.1        * Reference current source
VPLUS 2 0 DC 10        * Positive voltage supply
VMINUS 0 6 DC -10      * Negative voltage supply
VCE1 4 9 DC 5          * Collector-Emitter voltage for Q1
VCE2 7 6 DC 5          * Collector-Emitter voltage for Q2
VBE 4 6 DC 0.7         * Base-Emitter voltage for Q1 and Q2
VBE0 5 0 DC 0.7        * Base-Emitter voltage for Q0

Q1 4 4 9 NPN          * Transistor Q1 (C B E)
Q2 7 4 6 NPN          * Transistor Q2 (C B E)
Q0 5 4 7 NPN          * Transistor Q0 (C B E)

R1 3 4 1k             * Resistor R1
RC 2 5 1k             * Resistor RC

.END