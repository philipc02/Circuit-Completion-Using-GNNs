spice
* SPICE netlist for the given schematic

V1 5 3 DC 6V         * Voltage source between collector (5) and emitter (3)
V2 4 3 DC 3V         * Voltage source between ground (4) and emitter (3)

R1 4 2 43k           * Resistor connected to the base
R2 5 7 3.6k          * Resistor connected to the collector
R3 3 8 4.7k          * Resistor connected to the emitter

Q1 5 2 3 QNPN        * NPN transistor where 
                     * 5=Collector, 2=Base, 3=Emitter

.model QNPN NPN      * NPN transistor model

.end