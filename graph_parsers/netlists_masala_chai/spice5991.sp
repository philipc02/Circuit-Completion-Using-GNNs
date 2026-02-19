plaintext
* SPICE Netlist for the given schematic

Vsig 1 0 AC 1

Rsig 1 2 500k

I1 2 0 DC 200uA

Q1 5 2 3 NPN

I2 4 0 DC 200uA

Q2 4 8 6 NPN

Vcc 5 0 DC 5V

Vout 4 0

* Connections
* 1: Positive terminal of Vsig
* 2: Negative terminal of Vsig, Positive terminal of Rsig, Base of Q1
* 3: Emitter of Q1, Negative terminal of I1, Ground
* 4: Collector of Q2, Positive terminal of I2, Vo
* 5: Positive Supply Voltage, Collector of Q1
* 6: Emitter of Q2, Ground
* 8: Base of Q2

.end