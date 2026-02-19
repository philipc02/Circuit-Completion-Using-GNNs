spice
* SPICE Netlist for the Given Schematic

C1 4 5 C ; Capacitor C

L1 4 4 L ; Inductor L

Q1 4 3 2 NPN ; NPN Transistor with collector, base, emitter

V1 3 0 AC ; AC Voltage Source

RB 3 0 RB ; Resistor connected to base

C2 2 0 C2 ; Capacitor connected from emitter to ground

R1 2 0 R1 ; Resistor connected from emitter to ground

.END