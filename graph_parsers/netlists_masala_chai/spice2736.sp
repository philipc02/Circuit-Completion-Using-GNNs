plaintext
* SPICE Netlist

* Components
M1 Y Y 0 NMOS    * NMOS M1 (Drain=Y, Gate=Y, Source=0)
M2 X X 0 NMOS    * NMOS M2 (Drain=X, Gate=X, Source=0)

R1 4 X 1k        * Resistor R1 from Node 4 (VDD) to Node X

C1 X Y 1u        * Capacitor C1 from Node X to Node Y

* Voltage Source
VDD 4 0 DC 5V    * Voltage Source VDD connected to Node 4 (VDD) and GND

.END