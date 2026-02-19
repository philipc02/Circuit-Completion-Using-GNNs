spice
* SPICE Netlist for Given Circuit

L1 2 3 1uH        * Inductor L1
Rp 2 3 1k         * Resistor Rp

C1 2 4 1uF        * Capacitor C1
C2 4 3 1uF        * Capacitor C2

Vtest 5 0 DC 1V   * Voltage Source Vtest

I1 2 5 1mA        * Current Source gm*Vx

rx 5 2 1k         * Resistor rx