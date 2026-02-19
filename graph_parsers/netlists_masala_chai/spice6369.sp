plaintext
* SPICE Netlist

* Voltage Source
Vi 7 0 DC

* Capacitors
C1 2 22 1u
C2 3 4 1u

* Transistors
M1 22 2 6 6 NMOS
M2 2 22 6 6 NMOS

* Ideal Operational Amplifier
Eopamp 3 0 3 0 1

* Control Signals
Vph1 2 0 PULSE(0 5 0 10n 10n)
Vph2 22 0 PULSE(0 5 0 10n 10n)

* End of Netlist