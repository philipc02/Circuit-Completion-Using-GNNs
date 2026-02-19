* NMOS amplifier circuit

V1 5 6 DC vi        * Voltage source vi
M1 2 5 3 3 NMOS     * NMOS Transistor: Drain=2, Gate=5, Source=3, Body=3
RD 2 4 1k           * Resistor RD connected between nets 2 and 4
VSUPPLY 4 0 DC 0    * Output voltage connected to net 4

* Model Parameters for NMOS
.model NMOS nmos (level=1 VTO=1 KP=2e-5)

* Simulation Commands
.tran 1n 100n
.end