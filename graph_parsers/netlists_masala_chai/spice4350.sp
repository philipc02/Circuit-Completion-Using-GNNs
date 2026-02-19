plaintext
* SPICE Netlist for the given schematic

* Transistors
Q1 5 1 3 NMOS   * Drain, Gate, Source
Q2 6 2 3 NMOS   * Drain, Gate, Source
Q3 3 3 7 NMOS   * Drain, Gate, Source; IO connected, Source tied to V-
Q4 4 2 7 NMOS   * Drain, Gate, Source

* Capacitors
* Inductors

* Resistors
R1 9 4 1k       * Resistor connecting node 9 and node 4
RC1 5 19 1k     * Resistor connecting node 5 and node 19 (unspecified here)
RC2 6 19 1k     * Resistor connecting node 6 and node 19 (unspecified here)

* Voltage Sources
Vplus 19 0 DC V+ * Connected to a positive voltage node
Vminus 7 0 DC V- * Connected to a negative voltage node

* End of Netlist