plaintext
* SPICE Netlist

V1 3 0 DC 3V     * Voltage Source for Gate
V2 2 0 DC -3V    * Voltage Source for Drain

MPMOS 2 3 1 1 PMOS * PMOS Transistor (Drain, Gate, Source, Substrate)

RS 1 4 1k  * Source Resistor RS
RD 2 3 1k  * Drain Resistor RD

.model PMOS PMOS(L=1u W=1u)

.end