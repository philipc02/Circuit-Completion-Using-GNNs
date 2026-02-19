plaintext
* SPICE Netlist

* Voltage Source
V1 1 0 DC V_in

* Resistors
R1 1 3 R_prime
R2 2 4 R_prime
R3 3 2 R

* Capacitor
C1 2 5 C_value

* Operational Amplifier
* Assuming ideal Op-Amp
XU1 3 2 2 0 opamp

* Connections
* 1 -> Vin
* 2 -> Op-Amp inputs (inverting and non-inverting)
* 3 -> R and R' connections
* 4 -> Vout
* 5 -> Ground

.END