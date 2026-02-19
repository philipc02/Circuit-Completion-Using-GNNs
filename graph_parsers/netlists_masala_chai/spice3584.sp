plaintext
* SPICE netlist for the schematic

* Node Definitions
VCC 6 0 DC 5V
VIN 2 0 DC 4.4V
VREF 3 0 DC 4V

* Resistors
R1 5 6 210         * 210 Ω resistor
R2 4 6 235         * 235 Ω resistor
R3 3 0 1K          * 1 kΩ resistor

* BJTs
Q1 4 2 3 NPN       * NPN transistor
Q2 3 4 4 PNP       * PNP transistor (same as node collector)

* Current Sources
I1 5 4 DC 3.8mA    * Current source
I2 3 0 DC 3.8mA    * Current source

* End of netlist