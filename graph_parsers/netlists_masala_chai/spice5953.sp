plaintext
* SPICE Netlist for the given circuit

* Voltage Source
Vsig 9 0 DC <value> AC <AC value>

* Input Resistor
Rsig 5 4 <value>

* Transistor Q1
Q1 2 2 3 Q_Model

* Feedback Resistor for Q1
Rf1 0 2 <value>

* Transistor Q2
Q2 3 2 6 Q_Model

* Feedback Resistor for Q2
Rf2 3 7 <value>

* Load Resistor
RL 7 0 <value>

* Define the Q_Model for the BJTs
.model Q_Model NPN

* End of Netlist