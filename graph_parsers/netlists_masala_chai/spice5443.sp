spice
* SPICE netlist for the given circuit

V1 4 8 AC 1          * AC Voltage Source
R1 5 6 5.1k          * Resistor R1
Rp_eq 2 22 10.0k     * Resistor R_p(eq)
Leq 2 23 5.0mH       * Inductor L_eq
C1 3 4 0.022uF       * Capacitor C

* Connection to ground nodes
* Note: Ground node is conventionally '0' in SPICE.
0 8 0                * Connect node 8 to ground
0 5 0                * Connect node 5 to ground
0 22 0               * Connect node 22 to ground
0 23 0               * Connect node 23 to ground
0 3 0                * Connect node 3 to ground