plaintext
   * SPICE Netlist for the Schematic

   .MODEL NMOS NMOS LEVEL=1
   .MODEL PMOS PMOS LEVEL=1

   * Transistors
   M1 2 4 5 5 NMOS
   M3 5 9 0 0 NMOS
   M2 2 2 3 3 PMOS

   * Resistor
   RD 2 3 RD_VALUE

   * Voltage Sources
   VDD 3 0 DC VDD_VALUE
   VIN 4 0 DC VIN_VALUE

   .END