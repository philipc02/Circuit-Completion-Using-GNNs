* SPICE Netlist
* Capacitors
C1 4 5 6p
C2 3 0 7p
C3 2 0 7p

* Op Amp
* Note that op amp is modeled here with its connections
XOPAMP 5 7 4 opampmodel

* Define the op-amp model (for simplicity, using a generic model)
.model opampmodel opamp

.end