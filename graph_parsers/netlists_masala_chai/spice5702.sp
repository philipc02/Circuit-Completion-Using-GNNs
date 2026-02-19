spice
* NMOS Transistor
M1 2 5 3 3 NMOS

* Voltage Sources
VDD 6 2 DC <value_of_VDD>
VS 5 3 DC <value_of_VS>

* Define Nodes
* 1: ID (not used in circuit)
* 2: NMOS Drain
* 3: Ground
* 4: NMOS Source (also tied to body)
* 5: VS positive terminal
* 6: VDD

* Model Parameters
.model NMOS NMOS level=1